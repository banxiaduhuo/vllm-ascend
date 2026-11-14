# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for the scheduler-to-native recurrent address contract."""

from types import SimpleNamespace

import torch
import pytest

from vllm_ascend.xlite.recurrent_metadata import RecurrentStateMetadataConverter


def tensor(data):
    return torch.tensor(data, dtype=torch.int32)


def test_mixed_verify_spans_and_stable_addresses():
    converter = RecurrentStateMetadataConverter(8, 16, 4)
    meta = SimpleNamespace(
        spec_state_indices_tensor=tensor([[10, 11, 12], [20, 21, 22]]),
        non_spec_state_indices_tensor=tensor([30, 40]),
        spec_sequence_masks=torch.tensor([True, False, True, False]),
        num_accepted_tokens=tensor([1, 3]),
    )
    starts = tensor([0, 3, 5, 8, 8])
    reads, writes, conv = converter.convert([meta], starts, 4, 8)
    assert reads[0][:4].tolist() == [10, 30, 22, 40]
    assert writes[0][:8].tolist() == [10, 11, 12, 0, 30, 20, 21, 22]
    assert conv[0][:4].tolist() == [[10, 0, 5], [30, 0, 3], [20, 2, 5], [40, 0, 3]]
    addresses = [x[0].data_ptr() for x in (reads, writes, conv)]
    meta.spec_state_indices_tensor = None
    meta.non_spec_state_indices_tensor = tensor([50, 0])
    reads, writes, conv = converter.convert([meta], tensor([0, 2, 3]), 2, 3)
    assert [x[0].data_ptr() for x in (reads, writes, conv)] == addresses
    assert reads[0][:2].tolist() == [50, 0]
    assert writes[0][:3].tolist() == [0, 50, 0]
    assert conv[0][:2].tolist() == [[50, 0, 3], [0, 0, 3]]


def test_all_spec_null_pages_and_acceptance_offset():
    meta = SimpleNamespace(
        spec_state_indices_tensor=tensor([[0, 0], [4, 5]]),
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=torch.tensor([True, True]),
        num_accepted_tokens=tensor([1, 2]),
    )
    converter = RecurrentStateMetadataConverter(2, 4, 4)
    reads, writes, conv = converter.convert([meta, meta], tensor([0, 2, 4]), 2, 4)
    for group in range(2):
        assert reads[group].tolist() == [0, 5]
        assert writes[group].tolist() == [0, 0, 4, 5]
        assert conv[group].tolist() == [[0, 0, 4], [4, 1, 4]]


def test_recurrent_group_count_cannot_change_after_capture():
    meta = SimpleNamespace(spec_state_indices_tensor=None, non_spec_state_indices_tensor=tensor([1]))
    converter = RecurrentStateMetadataConverter(2, 4, 4)
    reads, _, _ = converter.convert([meta], tensor([0, 1]), 1, 1)
    address = reads[0].data_ptr()
    with pytest.raises(ValueError, match="group count changed"):
        converter.convert([meta, meta], tensor([0, 1]), 1, 1)
    assert reads[0].data_ptr() == address
