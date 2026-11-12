# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GLM recurrent operator followed by the graph-owned output writeback."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.glm5next.ops.kda import recurrent_kda


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("heads", [8, 32, 64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("padding", [0, 1])
@torch.inference_mode()
def test_recurrent_output_buffer_preserves_state_and_guards(heads, batch, padding):
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
    state = torch.zeros(batch + 2, heads, dim, dim, device="npu")
    indices = torch.arange(1, batch + 1, device="npu", dtype=torch.int32)[:, None]
    starts = torch.arange(batch + 1, device="npu", dtype=torch.int32)
    expected = recurrent_kda(q, k, v, gate, beta, state, starts, indices, a_log, dt_bias, -5.0)
    torch.npu.synchronize()
    expected_state = state.clone()
    state.zero_()
    output_tokens = tokens + 2
    storage = torch.full((output_tokens * heads * dim + 2 * guard,), -7.0, device="npu", dtype=q.dtype)
    destination = storage[guard:-guard].view(1, output_tokens, heads, dim)
    actual = recurrent_kda(q, k, v, gate, beta, state, starts, indices, a_log, dt_bias, -5.0, output_buffer=destination)
    torch.npu.synchronize()
    assert actual is destination
    torch.testing.assert_close(actual[:, :batch], expected[:, :batch], rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, batch:]) == 0
    assert torch.count_nonzero(state[0]) == 0
    assert torch.count_nonzero(state[-1]) == 0
    assert (storage[:guard] == -7).all()
    assert (storage[-guard:] == -7).all()
