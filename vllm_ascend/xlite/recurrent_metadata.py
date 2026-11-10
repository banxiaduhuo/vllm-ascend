# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Resolve scheduler recurrent state into capture-safe native address tables."""

from typing import Any

import torch


class RecurrentStateMetadataConverter:
    """Keep output addresses stable while resolving each batch entirely on device.

    Accepted counts include the sampled token (1 is the non-spec neutral value).
    Page zero is a null sentinel. Group order must match the runtime's linear
    layer round robin; the kpool tail table is not a recurrent group.
    """

    def __init__(self, max_batch_size: int, max_tokens: int, conv_kernel_dim: int):
        self._max_batch_size = max_batch_size
        self._max_tokens = max_tokens
        self._conv_rows = conv_kernel_dim - 1
        self._reads: list[torch.Tensor] = []
        self._writes: list[torch.Tensor] = []
        self._conv: list[torch.Tensor] = []

    def _allocate(self, num_groups: int, device: torch.device) -> None:
        self._reads = [
            torch.zeros(self._max_batch_size, dtype=torch.int32, device=device) for _ in range(num_groups)
        ]
        self._writes = [
            torch.zeros(self._max_tokens, dtype=torch.int32, device=device) for _ in range(num_groups)
        ]
        self._conv = [
            torch.zeros((self._max_batch_size, 3), dtype=torch.int32, device=device) for _ in range(num_groups)
        ]

    def convert(self, groups: list[Any], qsl: torch.Tensor, num_reqs: int, num_actual_tokens: int):
        if num_reqs > self._max_batch_size or num_actual_tokens > self._max_tokens:
            raise ValueError("xlite recurrent metadata exceeds configured capacity")
        if len(groups) != len(self._reads):
            self._allocate(len(groups), qsl.device)
        for gid, meta in enumerate(groups):
            self._fill_group(gid, meta, qsl, num_reqs, num_actual_tokens)
        return self._reads, self._writes, self._conv

    @staticmethod
    def _write_span_final_tokens(write, qsl, pages, nreq, ntok, spec_mask=None):
        tokens = torch.arange(ntok, device=qsl.device)
        req = torch.searchsorted(qsl[1:nreq], tokens.to(qsl.dtype), right=True).clamp(max=nreq - 1)
        final = tokens == qsl[req + 1] - 1
        if spec_mask is not None:
            final = final & ~spec_mask[req]
        write[:ntok].copy_(torch.where(final, pages[req], write[:ntok]))

    def _fill_group(self, gid, meta, qsl, nreq, ntok):
        spec = meta.spec_state_indices_tensor
        nonspec = meta.non_spec_state_indices_tensor
        if spec is None and nonspec is None:
            raise ValueError("xlite recurrent metadata requires state indices")
        read, write, locs = self._reads[gid], self._writes[gid], self._conv[gid]
        write[:ntok].zero_()
        if nreq == 0:
            return
        if spec is None or spec.shape[0] == 0:
            read[:nreq].copy_(nonspec[:nreq].reshape(-1))
            locs[:nreq, 0].copy_(read[:nreq])
            locs[:nreq, 1].zero_()
            locs[:nreq, 2].fill_(self._conv_rows)
            self._write_span_final_tokens(write, qsl, read, nreq, ntok)
            return
        mask = meta.spec_sequence_masks[:nreq]
        cols = spec.shape[1]
        expanded = torch.zeros((nreq, cols), dtype=torch.int32, device=qsl.device)
        expanded.masked_scatter_(mask[:, None].expand(-1, cols), spec.to(torch.int32))
        live = torch.zeros(nreq, dtype=torch.int32, device=qsl.device)
        if nonspec is not None:
            live.masked_scatter_(~mask, nonspec.to(torch.int32).reshape(-1))
        accepted = torch.ones(nreq, dtype=torch.int32, device=qsl.device)
        accepted.masked_scatter_(mask, meta.num_accepted_tokens.to(torch.int32))
        offset = (accepted - 1).clamp(0, cols - 1)
        read[:nreq].copy_(torch.where(mask, expanded.gather(1, offset.long()[:, None]).squeeze(1), live))
        locs[:nreq, 0].copy_(torch.where(mask, expanded[:, 0], live))
        locs[:nreq, 1].copy_(torch.where(mask, accepted - 1, 0))
        spans = qsl[1 : nreq + 1] - qsl[:nreq]
        locs[:nreq, 2].copy_(torch.where(mask, self._conv_rows + (spans - 1).clamp(min=0), self._conv_rows))
        tokens = torch.arange(ntok, device=qsl.device)
        req = torch.searchsorted(qsl[1:nreq], tokens.to(qsl.dtype), right=True).clamp(max=nreq - 1)
        col = (tokens - qsl[req]).clamp(0, cols - 1).long()
        write[:ntok].copy_(torch.where(mask[req], expanded[req, col], 0))
        self._write_span_final_tokens(write, qsl, live, nreq, ntok, mask)
