# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""GDN cache ownership and pitched pool pairing regression cases."""

import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.xlite.xlite import Qwen35XliteModel


class TestQwen35Cache(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(Qwen35XliteModel)
        attn = SimpleNamespace(
            prefix="model.layers.0.linear_attn",
            get_state_shape=lambda: ((4, 64), (1, 64, 64)),
            get_state_dtype=lambda: (torch.bfloat16, torch.float32),
        )
        self.adapter._get_layers_and_model_prefix = lambda: ([SimpleNamespace(linear_attn=attn)], "")
        self.adapter.xlite_config = SimpleNamespace(n_layers=1)
        self.adapter.full_attn_layers = []
        # Physical page padding must survive pairing unchanged.
        self.conv = torch.zeros(4, 5, 64, dtype=torch.bfloat16)[:, :4]
        self.ssm = torch.zeros(4, 1, 64, 64, dtype=torch.float32)

    def test_paired_split_and_aliases(self):
        for entries in (
            [(self.conv, self.ssm)],
            [self.conv, self.ssm],
            [self.ssm, self.conv],
            [(self.conv, self.ssm), (self.conv.view_as(self.conv), self.ssm.view_as(self.ssm))],
        ):
            with self.subTest(entries=len(entries)):
                conv, ssm = self.adapter._make_kda_layer_cache(0, entries)
                self.assertEqual(conv.data_ptr(), self.conv.data_ptr())
                self.assertEqual(conv.stride(), self.conv.stride())
                self.assertEqual(ssm.data_ptr(), self.ssm.data_ptr())

    def test_distinct_candidate_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            self.adapter._make_kda_layer_cache(0, [(self.conv, self.ssm), (self.conv.clone(), self.ssm.clone())])

    def test_exact_owner_excludes_other_module_with_same_layer_index(self):
        caches = {
            "model.layers.0.linear_attn": (self.conv, self.ssm),
            "other.layers.0.linear_attn": (self.conv.clone(), self.ssm.clone()),
        }
        (conv, ssm), = self.adapter.extract_kv_cache(caches)
        self.assertEqual(conv.data_ptr(), self.conv.data_ptr())
        self.assertEqual(ssm.data_ptr(), self.ssm.data_ptr())

    def test_state_descriptor_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conv pool"):
            self.adapter._make_kda_layer_cache(0, [self.conv[:, :3], self.ssm])


if __name__ == "__main__":
    unittest.main()
