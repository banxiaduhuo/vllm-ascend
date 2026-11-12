# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GLM recurrent operator followed by the graph-owned output writeback."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.glm5next.ops.kda import chunk_kda, recurrent_kda


def guarded_state(pages, heads, dim, dtype, paged=True):
    logical = heads * dim * dim
    pitch, offset = logical + (128 if paged else 0), 32
    raw = torch.full((pages * pitch + 64,), -7.0, device="npu", dtype=dtype)
    state = raw.as_strided((pages, heads, dim, dim), (pitch, dim * dim, dim, 1), offset)
    state.zero_()
    return raw, state


def assert_storage_matches(raw, state, before, expected):
    # Compare all logical pages AND prefix/suffix/inter-page guard elements.
    reference = before.clone()
    reference.as_strided(state.shape, state.stride(), state.storage_offset()).copy_(expected.cpu())
    torch.testing.assert_close(raw.cpu(), reference, rtol=0, atol=0)


def assert_guards_unchanged(raw, state, before):
    observed = raw.cpu().clone()
    # Ignore logical data here so numerical failures cannot hide a guard write.
    observed.as_strided(state.shape, state.stride(), state.storage_offset()).copy_(
        before.as_strided(state.shape, state.stride(), state.storage_offset())
    )
    torch.testing.assert_close(observed, before, rtol=0, atol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("heads", [8, 32, 64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_recurrent_output_buffer_preserves_state_and_guards(heads, batch, padding, paged, state_dtype):
    torch.manual_seed(42)
    dim, guard = 128, 32
    tokens = batch + padding
    merged = torch.randn(1, tokens, 3 * heads * dim, device="npu", dtype=torch.bfloat16) * 0.1
    # Preserve native merged-projection token strides instead of cloning Q/K/V.
    q, k, v = [part.view(1, tokens, heads, dim) for part in merged.split(heads * dim, dim=-1)]
    gate = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device="npu", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="npu")
    dt_bias = torch.zeros(heads * dim, device="npu")
    raw, state = guarded_state(batch + 2, heads, dim, state_dtype, paged)
    state[1 : batch + 1].normal_(std=0.01)
    before = raw.cpu().clone()
    # Independent compact baseline: testing two calls on the same pitched pool
    # would miss a shared stride bug in both calls.
    baseline = state.clone()
    indices = torch.arange(1, batch + 1, device="npu", dtype=torch.int32)[:, None]
    starts = torch.arange(batch + 1, device="npu", dtype=torch.int32)
    inputs_before = [x.clone() for x in (q, k, v, gate, beta)]
    expected = recurrent_kda(q, k, v, gate, beta, baseline, starts, indices, a_log, dt_bias, -5.0)
    for source, saved in zip((q, k, v, gate, beta), inputs_before):
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
    torch.npu.synchronize()
    output_tokens = tokens + 2
    storage = torch.full((output_tokens * heads * dim + 2 * guard,), -7.0, device="npu", dtype=q.dtype)
    destination = storage[guard:-guard].view(1, output_tokens, heads, dim)
    actual = recurrent_kda(q, k, v, gate, beta, state, starts, indices, a_log, dt_bias, -5.0, output_buffer=destination)
    torch.npu.synchronize()
    assert actual is destination
    assert_guards_unchanged(raw, state, before)
    assert (storage[:guard] == -7).all()
    assert (storage[-guard:] == -7).all()
    torch.testing.assert_close(actual[:, :batch], expected[:, :batch], rtol=0, atol=0)
    torch.testing.assert_close(state, baseline, rtol=0, atol=0)
    assert_storage_matches(raw, state, before, baseline)
    assert torch.count_nonzero(actual[:, batch:]) == 0
    assert torch.count_nonzero(state[0]) == 0
    assert torch.count_nonzero(state[-1]) == 0


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("heads", [8, 32, 64])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_initial", [False, True])
@torch.inference_mode()
def test_chunk_prefill_preserves_paged_state_and_guards(heads, state_dtype, has_initial):
    torch.manual_seed(42)
    dim, tokens = 128, 72
    raw, state = guarded_state(5, heads, dim, state_dtype)
    state.normal_(std=0.01)
    before = raw.cpu().clone()
    baseline = state.clone()
    q, k, v, gate = [torch.randn(1, tokens, heads, dim, device="npu", dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    beta = torch.randn(1, tokens, heads, device="npu", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="npu")
    dt_bias = torch.zeros(heads * dim, device="npu")
    indices = torch.tensor([3, 1], device="npu", dtype=torch.int32)
    flags = torch.tensor([has_initial, not has_initial], device="npu", dtype=torch.bool)
    metadata = SimpleNamespace(
        keep_meta=None,
        cu_seqlens_host=(0, 65, tokens),
        cu_seqlens_kern=None,
        chunk_indices_chunk64_host=(0, 0, 0, 1, 1, 0),
    )
    expected = chunk_kda(q, k, v, gate, beta, baseline, indices, flags, metadata, a_log, dt_bias, -5.0)
    actual = chunk_kda(q, k, v, gate, beta, state, indices, flags, metadata, a_log, dt_bias, -5.0)
    torch.npu.synchronize()
    assert_guards_unchanged(raw, state, before)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, baseline, rtol=0, atol=0)
    assert_storage_matches(raw, state, before, baseline)
    saved = before.as_strided(state.shape, state.stride(), state.storage_offset())
    torch.testing.assert_close(state[[0, 2, 4]].cpu(), saved[[0, 2, 4]], rtol=0, atol=0)
