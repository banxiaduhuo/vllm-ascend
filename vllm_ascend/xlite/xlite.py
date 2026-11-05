#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Xlite integration module for vLLM-Ascend."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import regex as re
import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.utils import extract_layer_index
from xlite._C import (
    AttnDSA,
    AttnHybridGdn,
    AttnHybridKda,
    AttnMetaV2,
    AttnMHA,
    AttnMLA,
    Runtime,
    ScoringFuncSigmoid,
    ScoringFuncSoftmax,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding
from vllm_ascend.xlite.utils import (
    _DUMMY_TENSOR,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    _TensorRepr,
    get_dotted_attr,
    get_layer_weights,
    is_dummy_tensor,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.v1.xlite.xlite_model_runner import XliteModelRunner as XliteModelRunnerV1
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerV2

    XliteModelRunner: TypeAlias = XliteModelRunnerV1 | XliteModelRunnerV2

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]


def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Cast a weight/bias tensor to fp32 (GLM5 indexer / MHC params are fp32-native)."""
    return t.to(torch.float32)


def _slice_rows(start: int, length: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Keep rows ``[start, start + length)`` in weights and their per-output-row quant tensors."""

    def apply(w: torch.Tensor) -> torch.Tensor:
        if w.ndim == 0 or w.numel() == 0:
            return w
        return w[start : start + length].contiguous()

    return apply


def _kv_cache_by_layer(kv_caches: dict[str, Any]) -> dict[int, list[Any]]:
    """Group the runner's ``name -> view`` mapping by the owning model layer.

    vLLM registers hybrid caches group-by-group, so the runner's flat list is not layer
    order. Names carry ``layers.<N>``; keep each layer's registered values (tensor or tuple)
    so adapters can pair them without relying on list position.
    """
    by_layer: defaultdict[int, list[Any]] = defaultdict(list)
    for name, cache in kv_caches.items():
        layer_id = extract_layer_index(name)
        if layer_id < 0:
            raise ValueError(f"xlite: cannot resolve the owning layer from KV cache name {name!r}.")
        by_layer[layer_id].append(cache)
    return dict(by_layer)


_architecture_strategy_map: dict[str, type[XliteModelBase]] = {}
"""Mapping from model architecture names in `config.json` to their corresponding xlite adapter classes."""


class XliteModelBase(ABC):
    """Base adapter for converting vLLM models into xlite runtime models.

    Subclasses are responsible for mapping architecture-specific configuration and weights into the `xlite._C.Model`
    interface.

    Attributes:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        npu_runnable (nn.Module): The original runnable model used by vLLM-ascend. Used as the source of truth for
            weight extraction for xlite model construction.
        vllm_config (VllmConfig): The configuration object provided by vLLM. Used to build xlite configuration at
            runtime.
        device (torch.device): The device on which the model is loaded. Typically an Ascend NPU device.
        dtype (torch.dtype): The data type used for model weights and computations. Derived from `vllm_config`.
        rottary_embed (AscendRotaryEmbedding | None): The rotary embedding module extracted from the NPU runnable model.
        xlite_config (XModelConfig): Native xlite configuration object populated by subclasses.
        xlite_model (XModel): Native xlite model container populated by subclasses.
        rope_cossin_cache (torch.Tensor): Precomputed RoPE frequency cache tensor for rotary positional embeddings.
    """

    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map."""
        arcs = getattr(cls, "_supported_architectures", None)
        if arcs is None:
            raise ValueError(f"XliteModel subclass {cls.__name__} must define _supported_architectures attribute.")
        if isinstance(arcs, str):
            arcs = [arcs]
        for arc in arcs:
            if arc in _architecture_strategy_map:
                raise ValueError(f"Duplicate xlite adapter for architecture {arc}: {_architecture_strategy_map[arc]}")
            _architecture_strategy_map[arc] = cls
        super().__init_subclass__(**kwargs)

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> None:
        """Initialize the xlite model adapter.

        Args:
            npu_runner (XliteModelRunner): The NPU model runner instance.
            vllm_config (VllmConfig): Runtime configuration used for model setup.

        Notes:
            The constructor stores the runnable model and vLLM config, and prepares empty xlite configuration and model
            containers for subclass-specific population.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.vllm_config = vllm_config
        self.device = self.npu_runner.device
        self.dtype = vllm_config.model_config.dtype
        # full mode: xlite handle all mixed and decode batches; otherwise, xlite only handle decode batches (roughly)
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.rottary_embed: AscendRotaryEmbedding | None = self._get_rope()
        if self.rottary_embed:
            logger.info_once(
                "xlite: found rotary embedding module `%s` in the NPU runnable.",
                re.sub(r"\s+", " ", repr(self.rottary_embed).replace("\n", "").strip()),
            )
        else:
            logger.warning_once("xlite: no rotary embedding module found in `%s`", self.npu_runnable)

        self.xlite_config = XModelConfig()
        self.xlite_model = XModel()
        self._build_model_config()
        self._build_model()
        self.xlite_model.init(self.xlite_config, torch.distributed.get_rank())
        self.rope_cossin_cache = self._precompute_rope_cache()

        if torch.distributed.get_rank() == 0:
            logger.info_once("xlite: built model config: %s", self.xlite_config)
            logger.info_once("xlite: built model weights: %s", self.xlite_model)

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]

    @abstractmethod
    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Extract xlite-compatible KV cache from the vLLM-ascend KV cache.

        Args:
            kv_caches (list[torch.Tensor]): The KV cache from the upstream vLLM/vLLM-ascend.

        Returns:
            list[tuple[torch.Tensor, ...]]: The KV cache in the format expected by xlite model implementations.
        """

    @abstractmethod
    def _build_model_config(self) -> None:
        """Build architecture-specific xlite model configuration.

        This method extracts necessary configuration attributes from the vLLM config (e.g., HuggingFace metadata) and
        populates an xlite :class:`ModelConfig` object.

        Returns:
            None: `self` attribute :attr:`xlite_config` is updated in-place.
        """

    @abstractmethod
    def _build_model(self) -> None:
        """Build architecture-specific xlite model weights.

        This method traverses the runnable model's parameters and maps them into the xlite :class:`Model` interface
        according to the architecture's specific structure.

        Returns:
            None: `self` attribute :attr:`xlite_model` is updated in-place.

        Notes:
            :meth:`_build_model_config` should be called prior to this method to ensure the xlite configuration is
            populated before weight mapping.
        """

    def _get_layers_and_model_prefix(self) -> tuple[Sequence[nn.Module], str]:
        """Extract transformer layers and parameter prefix from runnable.

        Returns:
            tuple[Sequence[nn.Module], str]: A pair of `(layers, model_prefix)` for model traversal.
        """
        if hasattr(self.npu_runnable, "language_model"):
            layers = cast(
                Sequence[nn.Module], get_dotted_attr(self.npu_runnable.language_model, "model.layers", default=[])
            )
            prefix = "language_model."
        else:
            layers = cast(Sequence[nn.Module], get_dotted_attr(self.npu_runnable, "model.layers", default=[]))
            prefix = ""
        return layers, prefix

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Extract the rotary embedding module from the NPU runnable model :data:`npu_runnable`.

        Args:
            layer_idx (int): The index of the transformer layer to extract the rotary embedding from. Defaults to 0.

        Returns:
            AscendRotaryEmbedding | None: The rotary embedding module found in the specified transformer layer, or None
            if not found.
        """
        layers, _ = self._get_layers_and_model_prefix()
        rottary_embed: AscendRotaryEmbedding | None = get_dotted_attr(layers, f"{layer_idx}.self_attn.rotary_emb")
        if not rottary_embed:
            logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return rottary_embed

    def _precompute_rope_cache(self) -> torch.Tensor:
        """Precomputes frequency-based complex exponential values for rotary positional embeddings (RoPE).

        This method generates the RoPE frequency cache (cosine and sine values) required by the xlite attention
        implementation. The cache should be precomputed on the NPU device to avoid unnecessary host-device transfers
        during inference.

        Returns:
            torch.Tensor: The precomputed RoPE frequency cache tensor ready for use in xlite attention computations.

        Notes:
            :meth:`_build_model_config` should be called prior to this method.
        """
        if not self.rottary_embed:
            return torch.empty(2, dtype=torch.float32, device=self.device)
        cossin_cache: torch.Tensor | None = getattr(self.rottary_embed, "cossin_cache", None)
        if not isinstance(cossin_cache, torch.Tensor):
            cossin_cache = self.rottary_embed._compute_cos_sin_cache()
        return cossin_cache.to(dtype=torch.float32, device=self.device)

    @staticmethod
    def is_tensor_nz(t: torch.Tensor) -> bool:
        """Check if a tensor is in NZ format.

        Args:
            t (torch.Tensor): The tensor to check.

        Returns:
            bool: True if the tensor is in NZ format, False otherwise.
        """
        format = torch_npu.get_npu_format(t)  # type: ignore[attr-defined]
        return format == torch_npu.Format.FRACTAL_NZ  # type: ignore[attr-defined]

    @staticmethod
    def all_tensors_zero(tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None) -> bool:
        """Check if all tensors in the list/tuple are zero tensors.

        Args:
            tensors (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None): The tensors to check.

        Returns:
            bool: True if all tensors are zero tensors (or empty), False otherwise.
        """
        if tensors is None:
            return True
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        if len(tensors) == 0:
            return True
        return all(torch.allclose(t, t.new_zeros(1)) for t in tensors)

    @staticmethod
    def _transform_deq_scale(deq_scale: torch.Tensor) -> torch.Tensor:
        """Repack a dequantization scale into the fixpipe hardware's expected format.

        Data is stored in ``uint64_t``: the upper 32 bits are 0 and the lower 32 bits hold an FP32 value whose lower 10
        bits are not involved in computation, making the effective data format TF32.

        Args:
            deq_scale (torch.Tensor): The original dequantization scale tensor.

        Returns:
            torch.Tensor: The repacked scale tensor for fixpipe computation.
        """
        if deq_scale.dtype == torch.int64:
            return deq_scale
        deq_scale_fp32 = deq_scale.to(torch.float32)
        scale = deq_scale_fp32.new_zeros(deq_scale.shape[0] * 2)
        scale[0::2] = deq_scale_fp32[0::1]
        return scale

    @property
    def hf_text_config(self) -> PretrainedConfig:
        """Convenience property to access HuggingFace text configuration from vLLM config.

        Returns:
            PretrainedConfig: The HuggingFace text configuration object extracted from vLLM config.
        """
        hf_config = self.vllm_config.model_config.hf_text_config
        return cast(PretrainedConfig, getattr(hf_config, "text_config", hf_config))

    @property
    def hf_vision_config(self) -> PretrainedConfig | None:
        """Convenience property to access HuggingFace vision configuration from vLLM config, if exists.

        Returns:
            PretrainedConfig | None: The HuggingFace vision configuration object extracted from vLLM config, or None if
                not present.
        """
        return getattr(self.vllm_config.model_config.hf_config, "vision_config", None)


class StandardXliteModel(XliteModelBase):
    """xlite adapter base for standard architectures.

    This is the *de facto* base adapter for all xlite-supported architectures. Configurations for various model types,
    dense and MoE, are included.

    Subclasses of :class:`XliteModelBase`, i.e., xlite model adapters, should inherit from :class:`StandardXliteModel`,
    unless there is a major divergence.

    **Developer Guide**: The :class:`StandardXliteModel` class is designed to be a flexible base for xlite adapters.
    When developing a new adapter for a specific architecture to dock with the xlite backend, consider adding weight
    loading lines in :meth:`StandardXliteModel._build_model` directly if feasible, using the highly flexible and
    error-checked `get_layer_weights` utility. For :meth:`_build_model_config`, override it in the subclass only if the
    architecture has unique configuration needs.
    """

    _supported_architectures = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3VLForConditionalGeneration",
    ]

    def _pin_cache_geometry(self, backends: Sequence[type]) -> int:
        """Pin kernel page geometry before Model.init() copies the config.

        ``cache_config.block_size`` is only final after the platform's
        update_block_size_for_backend(); executor init runs it, but its
        hybrid/dtype alignment phases may still move it past any earlier
        snapshot, so re-run it here and pick the kernel page every backend
        in ``backends`` supports as a factor of the manager page. Rounding
        max_seq_len up to whole manager pages keeps the native block-table
        rows page-complete; the planner then budgets the final page's unused
        columns while vLLM keeps its requested model-length limit.

        Returns:
            int: the kernel block size, so subclasses can assemble family
            ``block_sizes`` vectors (e.g. GLM5's [main, pooled, tail-ring]).
        """
        from vllm.platforms import current_platform
        from vllm.utils.math_utils import cdiv
        from vllm.v1.worker.utils import select_common_block_size

        current_platform.update_block_size_for_backend(self.vllm_config)
        manager_block_size = self.vllm_config.cache_config.block_size
        kernel_block_size = select_common_block_size(manager_block_size, backends)
        self.xlite_config.block_size = kernel_block_size
        self.xlite_config.max_seq_len = cdiv(self.xlite_config.max_seq_len, manager_block_size) * manager_block_size
        return kernel_block_size

    def _build_model_config(self) -> None:
        xlite_config, vllm_config, hf_config = self.xlite_config, self.vllm_config, self.hf_text_config

        xlite_config.vocab_size = hf_config.vocab_size
        xlite_config.hidden_size = hf_config.hidden_size
        xlite_config.n_layers = hf_config.num_hidden_layers
        xlite_config.n_heads = hf_config.num_attention_heads
        xlite_config.n_kv_heads = hf_config.num_key_value_heads
        if hasattr(hf_config, "head_dim"):
            xlite_config.head_dim = hf_config.head_dim
        else:
            xlite_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        xlite_config.rope_head_dim = self.rottary_embed.rotary_dim if self.rottary_embed else xlite_config.head_dim
        xlite_config.norm_eps = hf_config.rms_norm_eps
        if hasattr(hf_config, "rope_theta"):
            xlite_config.rope_theta = hf_config.rope_theta
        else:
            xlite_config.rope_theta = getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
        xlite_config.softmax_scale = xlite_config.head_dim**-0.5
        xlite_config.n_dense_layers = hf_config.num_hidden_layers
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.def_tp_size = tp_size = get_tensor_model_parallel_world_size()
        xlite_config.def_dp_size = vllm_config.parallel_config.data_parallel_size
        try:
            ep_word_size = get_ep_group().world_size
            xlite_config.moe_ep_size = ep_word_size if vllm_config.parallel_config.enable_expert_parallel else 1
            xlite_config.moe_tp_size = 1 if vllm_config.parallel_config.enable_expert_parallel else ep_word_size
        except AssertionError:
            xlite_config.moe_ep_size, xlite_config.moe_tp_size = 1, 1
        xlite_config.experts_weight_transpose = True

        xlite_config.attn_type = AttnMHA
        xlite_config.scoring_func = ScoringFuncSoftmax
        xlite_config.weight_nz = get_ascend_config().weight_nz_mode == 2
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        # configure runtime token budgets between xlite full/decode-only modes with respect to speculative decoding
        # NOTE: for profiling runs (e.g., graph capturing), the following configurations might need further adjustments
        scheduler_config, spec_config = vllm_config.scheduler_config, vllm_config.speculative_config
        max_num_active_seqs = getattr(scheduler_config, "max_num_active_seqs", None) or scheduler_config.max_num_seqs
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.num_spec_tokens = num_spec_tokens = spec_config.num_speculative_tokens if spec_config else 0
        draft_slots = spec_config.max_num_new_slots_for_drafting if spec_config else 0
        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens or max_num_batched_tokens
        max_num_scheduled_tokens = min(max_num_scheduled_tokens, max_num_batched_tokens)
        if get_ascend_config().xlite_graph_config.full_mode:
            max_m = max_num_batched_tokens  # all request must confine to this budget per forward pass
        else:
            # xlite only handles decode batches - drafter's additional slots consume the same scheduler input budget
            max_m = max(
                min(n * (num_spec_tokens + 1), max_num_scheduled_tokens, max_num_batched_tokens - n * draft_slots)
                for n in range(1, max_num_active_seqs + 1)
            )
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

    def _bind_embed_norm_head(
        self, model_prefix: str, norm_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    ) -> None:
        """Bind the token embedding, final RMSNorm and (tied or separate) lm_head.

        ``norm_fn`` lets Gemma-style adapters convert ``x * (1 + w)`` to ``x * w`` once here
        instead of duplicating the conversion at each adapter's call site.
        """
        xlite_model = self.xlite_model
        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        xlite_model.norm = norm_fn(norm) if norm_fn is not None else norm
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

    def _bind_layer_norms(
        self, layers: Sequence[nn.Module], norm_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    ) -> None:
        """Bind per-layer input/post-attention norms (``norm_fn`` for Gemma-style x*(1+w))."""
        attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")
        if norm_fn is not None:
            attn_norm = [norm_fn(w) for w in attn_norm]
            mlp_norm = [norm_fn(w) for w in mlp_norm]
        self.xlite_model.attn_norm = attn_norm
        self.xlite_model.mlp_norm = mlp_norm

    def _bind_dense_mlp(self, layers: Sequence[nn.Module]) -> None:
        """Bind the dense MLP slots (MoE layers carry no gate_up_proj and are skipped)."""
        mlp_prefix = self._decoder_layer_mlp_module
        self.init_matmul_weights(layers, "mlp_up_gate", f"{mlp_prefix}.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", f"{mlp_prefix}.down_proj")

    def init_matmul_weights(
        self,
        layers: Sequence[torch.nn.Module],
        xlite_prefix: str,
        model_prefix: str,
        *,
        mask: Sequence[bool] | None = None,
        row_processor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
            row_processor (Callable | None): Optional row-wise transform for fused/sliced projections.
                It is applied to the weight and to per-output-row quant tensors (deq/weight scale,
                quant bias), but not to activation-side scales.
        """
        xlite_model = self.xlite_model

        def maybe_rows(tensor: torch.Tensor) -> torch.Tensor:
            return row_processor(tensor) if row_processor is not None else tensor

        setattr(
            xlite_model,
            xlite_prefix,
            get_layer_weights(layers, f"{model_prefix}.weight", mask=mask, post_processor=maybe_rows),
        )
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str, *, rows: bool = False) -> None:
            post_processor = maybe_rows if rows else None
            setattr(
                xlite_model,
                xlite_attr,
                get_layer_weights(layers, layer_attr, mask=mask, post_processor=post_processor),
            )

        def deq_processor(tensor: torch.Tensor) -> torch.Tensor:
            return self._transform_deq_scale(maybe_rows(tensor))

        # A masked probe is polluted by dummy placeholders for missing attrs (they are
        # rebindable non-empty tensors), which would misclassify dynamic-quant
        # checkpoints as static. Probe unmasked first: missing attrs filter out to an
        # empty list, so a non-empty result is a reliable static-quant signal.
        if len(get_layer_weights(layers, f"{model_prefix}.deq_scale")) > 0:  # static quant
            deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", mask=mask, post_processor=deq_processor)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias", rows=True)
        else:
            weight_scale = get_layer_weights(
                layers, f"{model_prefix}.weight_scale", mask=mask, post_processor=deq_processor
            )
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)

    def _build_model(self) -> None:
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()

        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)

        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj")
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj")

        with xlite_model.condition(lambda tensors: len(tensors) == xlite_config.n_layers):
            xlite_model.mha_q_norm = get_layer_weights(layers, "self_attn.q_norm.weight")
            xlite_model.mha_k_norm = get_layer_weights(layers, "self_attn.k_norm.weight")
            xlite_config.qk_norm = bool(xlite_model.mha_q_norm) and bool(xlite_model.mha_k_norm)

        # Dense MLP weights
        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        # MoE-specific weights
        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0]) if xlite_model.gate else False
        xlite_model.gate_bias = get_layer_weights(
            layers,
            f"{mlp_prefix}.gate.e_score_correction_bias",
            f"{mlp_prefix}.e_score_correction_bias",  # for MiniMax-2.x compatibility
            post_processor=lambda b: b.to(torch.float32),  # type conversion for numerical stability in xlite backend
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")

        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])

        # bias terms
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mha_qkv_bias = get_layer_weights(layers, "self_attn.qkv_proj.bias")
            xlite_config.qkv_bias = len(xlite_model.mha_qkv_bias) == xlite_config.n_layers
            xlite_model.norm_bias = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.bias", raises=True)
            xlite_model.attn_norm_bias = get_layer_weights(layers, "input_layernorm.bias")
            xlite_model.mlp_norm_bias = get_layer_weights(layers, "post_attention_layernorm.bias")
            if xlite_config.qk_norm:
                xlite_model.mha_q_norm_bias = get_layer_weights(layers, "self_attn.q_norm.bias")
                xlite_model.mha_k_norm_bias = get_layer_weights(layers, "self_attn.k_norm.bias")

        if not self.quantization:
            return

        xlite_config.quant_attn_weight_nz = bool(xlite_model.mha_qkv) and self.is_tensor_nz(xlite_model.mha_qkv[0])
        xlite_config.quant_attn_weight_transpose = bool(xlite_model.mha_qkv)

        xlite_model.re_up_gate_scale_bias = get_layer_weights(layers, f"{re_prefix}.w13_scale_bias", **re_kwargs)
        xlite_model.re_down_scale_bias = get_layer_weights(layers, f"{re_prefix}.w2_scale_bias", **re_kwargs)

        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def init_matmul_weights(
        self,
        layers: Sequence[torch.nn.Module],
        xlite_prefix: str,
        model_prefix: str,
        *,
        mask: Sequence[bool] | None = None,
        row_processor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
            row_processor (Callable | None): Optional row-wise transform for fused/sliced projections.
                It is applied to the weight and to per-output-row quant tensors (deq/weight scale,
                quant bias), but not to activation-side scales.
        """
        xlite_model = self.xlite_model

        def maybe_rows(tensor: torch.Tensor) -> torch.Tensor:
            return row_processor(tensor) if row_processor is not None else tensor

        setattr(
            xlite_model,
            xlite_prefix,
            get_layer_weights(layers, f"{model_prefix}.weight", mask=mask, post_processor=maybe_rows),
        )
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str, *, rows: bool = False) -> None:
            post_processor = maybe_rows if rows else None
            setattr(
                xlite_model,
                xlite_attr,
                get_layer_weights(layers, layer_attr, mask=mask, post_processor=post_processor),
            )

        def deq_processor(tensor: torch.Tensor) -> torch.Tensor:
            return self._transform_deq_scale(maybe_rows(tensor))

        wt_kwargs: WeightGetterConfig = {"mask": mask, "post_processor": deq_processor}
        # DSA's indexer has shared layers (no own weights), so `mask` pads them with `_DUMMY_TENSOR`; the list is
        # thus never empty and a naive `len() > 0` check would misclassify dynamic-quant matmuls as static. Static
        # quant is when `aclnn_input_scale_reciprocal` exists and at least one entry is a real (non-dummy) tensor.
        input_scale = get_layer_weights(layers, f"{model_prefix}.aclnn_input_scale_reciprocal", mask=mask)
        is_static = any(not is_dummy_tensor(t) for t in input_scale)
        if is_static:
            setattr(xlite_model, f"{xlite_prefix}_input_scale", input_scale)
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias", rows=True)
            deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
        else:
            weight_scale = get_layer_weights(layers, f"{model_prefix}.weight_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", weight_scale)
    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove the type conversion once `xlite` accepts and defaults the cache in FP32 format
        return super()._precompute_rope_cache().to(dtype=self.dtype, device=self.device)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        kv_caches = kv_caches[: 2 * self.xlite_config.n_layers]
        if wrong_dims := set(cache.shape[-1] for cache in kv_caches) - set([self.xlite_config.head_dim]):
            raise ValueError(f"xlite: kv_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        return [(k_cache, v_cache) for k_cache, v_cache in zip(kv_caches[0::2], kv_caches[1::2])]


class QwenMoeXliteModel(StandardXliteModel):
    """xlite adapter for Qwen MoE architectures."""

    _supported_architectures = ["Qwen3MoeForCausalLM", "Qwen3VLMoeForConditionalGeneration"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob


class Glm4MoeXliteModel(StandardXliteModel):
    """xlite adapter for GLM4 MoE architectures."""

    _supported_architectures = ["Glm4MoeForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.norm_topk_prob = hf_config.norm_topk_prob
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = hf_config.routed_scaling_factor


class HybridXliteModel(StandardXliteModel):
    """Base for hybrid KDA (linear attention) + full-attention adapters (Qwen3.5 / GLM5).

    Owns the shared model-loading framework (embed/norm/head, layer norms, KDA-vs-full layer
    split, slot hooks) and the metadata/cache plumbing (AttnMetaV2 family tables built from the
    runner's raw persistent block tables, plus the structured per-layer KV cache mapping).
    """

    _supported_architectures: list[str] = []  # abstract base: not a concrete adapter
    _structured_kv_cache = True
    _layer_major_kv_cache = True

    # ---- model loading ---------------------------------------------------------------------

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Scan forward: hybrid models may start with a KDA layer that has no ``rotary_emb``."""
        layers, _ = self._get_layers_and_model_prefix()
        for i in range(layer_idx, len(layers)):
            if (rottary_embed := get_dotted_attr(layers, f"{i}.self_attn.rotary_emb")) is not None:
                return rottary_embed
        logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return None

    def _norm_fn(self) -> Callable[[torch.Tensor], torch.Tensor] | None:
        """Layer-norm transform hook (Gemma-style adapters return :meth:`_gemma_norm`)."""
        return None

    def _init_hybrid_layers(self, layers: Sequence[nn.Module]) -> list[int]:
        """Return the KDA (linear-attention) layer indices for this model."""
        raise NotImplementedError

    def _bind_kda_layer_weights(self, layers: Sequence[nn.Module], kda_mask: Sequence[bool]) -> None:
        """Bind the KDA/linear-attention slots (``kda_mask`` marks KDA layers)."""
        raise NotImplementedError

    def _bind_full_layer_weights(self, layers: Sequence[nn.Module], full_mask: Sequence[bool]) -> None:
        """Bind the full-attention slots (``full_mask`` marks full-attention layers)."""
        raise NotImplementedError

    def _bind_shared_weights(self, layers: Sequence[nn.Module]) -> None:
        """Bind model-shared slots (dense MLP, MoE, MHC, quant scales); optional."""

    def _build_model(self) -> None:
        layers, model_prefix = self._get_layers_and_model_prefix()
        norm_fn = self._norm_fn()
        self._bind_embed_norm_head(model_prefix, norm_fn)
        self._bind_layer_norms(layers, norm_fn)
        self.kda_layers = self._init_hybrid_layers(layers)
        kda_set = set(self.kda_layers)
        self.full_attn_layers = [i for i in range(self.xlite_config.n_layers) if i not in kda_set]
        # C++ DeriveLayerTypes/_kv cache slots key off these per-layer lists.
        self.xlite_config.kda_layers = list(self.kda_layers)
        self.xlite_config.full_attn_layers = list(self.full_attn_layers)
        kda_mask = [i in kda_set for i in range(len(layers))]
        full_mask = [not m for m in kda_mask]
        self._bind_kda_layer_weights(layers, kda_mask)
        self._bind_full_layer_weights(layers, full_mask)
        self._bind_shared_weights(layers)

    def _set_kda_config(self, k_heads: int, v_heads: int, k_dim: int, v_dim: int, conv_dim: int) -> None:
        cfg = self.xlite_config
        cfg.linear_num_k_heads = k_heads
        cfg.linear_num_v_heads = v_heads
        cfg.linear_key_head_dim = k_dim
        cfg.linear_value_head_dim = v_dim
        cfg.linear_conv_kernel_dim = conv_dim

    # ---- attention metadata ----------------------------------------------------------------

    @staticmethod
    def _group_id(kv_cache_groups: Sequence[Any], group: Any) -> int:
        for gid, candidate in enumerate(kv_cache_groups):
            if candidate is group:
                return gid
        raise ValueError("xlite hybrid: KV cache group not found")

    def _main_attn_group(self, kv_cache_groups: Sequence[Any]) -> Any:
        """Return the KV cache group that owns the full-attention KV cache."""
        raise NotImplementedError

    def _extra_attn_groups(self, kv_cache_groups: Sequence[Any]) -> Sequence[Any]:
        """Return the extra block-table family groups (mamba, and GLM5's kpool tail)."""
        raise NotImplementedError

    def select_main_common_group(self, by_group: dict[int, Any], kv_cache_groups: Sequence[Any]) -> Any:
        """Pick the main full-attention group's common metadata (group 0 may be a KDA group)."""
        return by_group[self._group_id(kv_cache_groups, self._main_attn_group(kv_cache_groups))]

    def augment_attn_meta(
        self,
        meta: AttnMetaV2,
        by_group: dict[int, Any],
        kv_cache_groups: Sequence[Any],
        is_dummy: bool,
    ) -> None:
        """Append the extra family block tables to the main-KV slot, verbatim."""
        tables = []
        for group in self._extra_attn_groups(kv_cache_groups):
            tables.append(
                torch.zeros((1, 1), dtype=torch.int32, device=self.device)
                if is_dummy
                else by_group[self._group_id(kv_cache_groups, group)].block_table_tensor
            )
        # Rebuild from the common main-KV slot so repeated dummy runs stay idempotent.
        meta.block_tables = [*meta.block_tables[:1], *tables]

    # ---- structured KV cache ---------------------------------------------------------------

    @staticmethod
    def _unwrap_cache_entry(entry: Any) -> Any:
        """Unwrap the runner's one-element tuple/list cache-view wrappers."""
        while isinstance(entry, (list, tuple)) and len(entry) == 1:
            entry = entry[0]
        return entry

    @staticmethod
    def _expect_cache_shape(cache: torch.Tensor, tail: Sequence[int], name: str, layer: int) -> torch.Tensor:
        """Validate a runner cache view of shape ``[slots, *tail]``."""
        if cache.dim() != len(tail) + 1 or tuple(cache.shape[1:]) != tuple(tail):
            raise ValueError(
                f"xlite hybrid: layer {layer} {name} {tuple(cache.shape)} does not match [slots, {tuple(tail)}]"
            )
        return cache

    def _kv_cache_entry_count(self, layer: int) -> int:
        """Number of runner cache entries that belong to model ``layer``."""
        raise NotImplementedError

    def _make_layer_cache(self, layer: int, entries: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        """Build one layer's xlite family tuple from its runner cache entries."""
        raise NotImplementedError

    def extract_kv_cache(self, kv_caches: list[Any] | dict[str, Any], /) -> list[tuple[torch.Tensor, ...]]:
        """Map the runner's cache entries to per-layer family tuples.

        V2 hands the ``name -> view`` mapping; resolve ownership from the name (`layers.<N>`) so
        group-by-group registration cannot swap hybrid states. A flat list remains supported.
        """
        n_layers = self.xlite_config.n_layers
        if isinstance(kv_caches, dict):
            by_layer = _kv_cache_by_layer(kv_caches)
            result: list[tuple[torch.Tensor, ...]] = []
            for layer in range(n_layers):
                entries = by_layer.get(layer)
                if not entries:
                    raise ValueError(f"xlite hybrid: no KV cache registered for layer {layer}.")
                result.append(self._make_layer_cache(layer, entries))
            logger.info_once("xlite hybrid: registered kv caches: %s", _TensorRepr(result, max_it=None))
            return result

        expected = sum(self._kv_cache_entry_count(layer) for layer in range(n_layers))
        if len(kv_caches) < expected:
            raise ValueError(f"xlite hybrid: kv cache count mismatch ({len(kv_caches)} vs. {expected}).")

        result = []
        cursor = 0
        for layer in range(n_layers):
            count = self._kv_cache_entry_count(layer)
            result.append(self._make_layer_cache(layer, kv_caches[cursor : cursor + count]))
            cursor += count
        logger.info_once("xlite hybrid: registered kv caches: %s", _TensorRepr(result, max_it=None))
        return result


_hybrid_common_md_bridge_installed = False


def install_hybrid_common_metadata_bridge() -> None:
    """Wire the V2 per-group common-metadata channel into the hybrid model state.

    基线(48ae9acde)只为 default state 接了 common_metadata_callback(见基座
    default.py prepare_attn 中 `xlite_graph_config.enabled` 分支),而
    AscendMambaHybridModelState.prepare_attn 不注册——hybrid attention 是本
    适配首个使用者。这里按 default 相同语义补上生产者:每步重建 per-group
    表并把逐组 callback 注入 build_attn_metadata;消费端(_common_attn_metadata_by_group /
    select_main_common_group)无需改动。基座将来补上同样接线后可直接删除本桥接。
    """
    global _hybrid_common_md_bridge_installed
    if _hybrid_common_md_bridge_installed:
        return
    from vllm_ascend.worker.v2.model_states import mamba_hybrid as mh

    orig_prepare_attn = mh.AscendMambaHybridModelState.prepare_attn
    orig_build_attn_metadata = mh.build_attn_metadata
    # V2 worker 单线程串行 prepare,模块级信封把 model_state 实例接给 build 包装层。
    building: list[Any] = []

    def prepare_attn(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # 对齐 default.py:每步重建 per-group common 表(避免跨步引用陈旧 metadata)。
        self._decoder_replay_common_metadata = {}
        building.append(self)
        try:
            return orig_prepare_attn(self, *args, **kwargs)
        finally:
            building.clear()

    def build_attn_metadata(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if building and kwargs.get("common_metadata_callback") is None:
            sink = building[0]._decoder_replay_common_metadata
            kwargs["common_metadata_callback"] = lambda group_id, common: sink.__setitem__(group_id, common)
        return orig_build_attn_metadata(*args, **kwargs)

    mh.AscendMambaHybridModelState.prepare_attn = prepare_attn
    mh.build_attn_metadata = build_attn_metadata  # _build_attn_metadata_for_batch 走模块全局名,重绑即生效
    _hybrid_common_md_bridge_installed = True
    logger.info_once("xlite: hybrid common-metadata bridge installed")


class Qwen35XliteModel(HybridXliteModel):
    """xlite adapter for Qwen3.5: hybrid GatedDeltaNet (KDA) linear attention + GQA full attention.

    Binds the vLLM-sharded weights to the xlite layouts: Gemma-style RMSNorm ``x * (1 + w)`` is
    converted to the standard ``x * w`` (:meth:`_gemma_norm`); the full-attention ``qkv_proj``
    rows ``[q|gate interleaved, k, v]`` are repacked to ``[Q|K|V|Gate]`` (``attn_output_gate``);
    the fused GDN ``in_proj_qkvz``/``in_proj_ba`` rows ``[q|k|v|z]``/``[b|a]`` are split into the
    C++ ``linear_in_proj_qkv/z/b/a`` slots.

    Metadata is assembled into :class:`AttnMetaV2` with ``block_tables = [main paged KV, mamba
    state indices]`` and a single main ``slot_mapping``.
    """

    # "Qwen3_5*ForConditionalGeneration": the open-weight checkpoints are the VLM-wrapped
    # architectures (text_config + vision tower); text layers resolve via the shared
    # language_model.* prefix path and the ViT stays on the native inputs_embeds route.
    _supported_architectures = [
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ]

    def _build_model_config(self) -> None:
        # Qwen3.5 MoE text configs carry only moe_intermediate_size (no dense FFN field); fill
        # the slot before the base builder reads it -- unused when first_k_dense_replace == 0,
        # i.e. every layer routes through the experts.
        if not hasattr(self.hf_text_config, "intermediate_size"):
            self.hf_text_config.intermediate_size = self.hf_text_config.moe_intermediate_size
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        if getattr(self.vllm_config.parallel_config, "prefill_context_parallel_size", 1) > 1:
            raise NotImplementedError("xlite qwen3.5: prefill context parallelism is not supported.")

        layers, _ = self._get_layers_and_model_prefix()
        if any(getattr(layer, "layer_scale", False) for layer in layers):
            raise NotImplementedError("xlite qwen3.5: layer_scale checkpoints are not supported.")
        full_attn = next(
            (layer.self_attn for layer in layers if getattr(layer, "layer_type", "") == "full_attention"), None
        )
        if full_attn is None:
            raise ValueError("xlite qwen3.5: no full-attention layer found in the runnable model.")

        # Hybrid dispatch (KDA linear attention + GQA full attention).
        xlite_config.attn_type = AttnHybridGdn
        xlite_config.full_attention_interval = getattr(hf_config, "full_attention_interval", 4)
        self._set_kda_config(
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
        )
        # Derive from the native attention module instead of assuming every Qwen3.5 is gated.
        xlite_config.attn_output_gate = bool(getattr(full_attn, "attn_output_gate", True))
        xlite_config.qk_norm = (
            getattr(full_attn, "q_norm", None) is not None and getattr(full_attn, "k_norm", None) is not None
        )
        # qkv bias follows the actual weights (the vllm Qwen3Next qkv_proj is built with bias=False).
        xlite_config.qkv_bias = False
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        xlite_config.block_sizes = [self._pin_cache_geometry([AscendAttentionBackend])]

        # MoE (Qwen3_5MoeForCausalLM only; the dense model keeps the Standard all-dense defaults).
        if getattr(hf_config, "num_experts", 0):
            xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
            xlite_config.n_routed_experts = hf_config.num_experts
            xlite_config.n_act_experts = hf_config.num_experts_per_tok
            xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
            xlite_config.norm_topk_prob = getattr(hf_config, "norm_topk_prob", True)
            xlite_config.scoring_func = ScoringFuncSoftmax
            xlite_config.route_scale = 1.0

    @staticmethod
    def _gemma_norm(weight: torch.Tensor) -> torch.Tensor:
        """Gemma RMSNorm ``x * (1 + w)`` -> standard RMSNorm ``x * w`` for the xlite kernels."""
        return (weight.float() + 1.0).to(weight.dtype).contiguous()

    def _norm_fn(self) -> Callable[[torch.Tensor], torch.Tensor] | None:
        # All Qwen3.5 RMSNorms are Gemma-style: convert x*(1+w) -> x*w.
        return self._gemma_norm

    def _init_hybrid_layers(self, layers: Sequence[nn.Module]) -> list[int]:
        # Per-layer KDA vs full-attention dispatch from the runnable layer modules (authoritative;
        # layer_type is set on every Qwen3_5DecoderLayer).
        return [i for i, layer in enumerate(layers) if getattr(layer, "layer_type", "") == "linear_attention"]

    def _bind_full_layer_weights(self, layers: Sequence[nn.Module], full_mask: Sequence[bool]) -> None:
        """GQA weights: repack qkv_proj ``[q|gate interleaved, k, v]`` to ``[Q|K|V|Gate]``."""
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        hf_config = self.hf_text_config
        full_attn = next(
            (layer.self_attn for layer in layers if getattr(layer, "layer_type", "") == "full_attention"), None
        )
        head_dim = int(getattr(full_attn, "head_dim", xlite_config.head_dim))
        # Prefer the native module's TP-local kv size over re-deriving it from the HF config.
        kv_size = (
            int(getattr(full_attn, "kv_size", 0))
            or max(1, hf_config.num_key_value_heads // xlite_config.def_tp_size) * head_dim
        )

        def _repack_qkv(w: torch.Tensor) -> torch.Tensor:
            """Repack ``[q|gate interleaved, k, v]`` rows; also used for 1-D quant side tensors."""
            if w.numel() == 0 or not xlite_config.attn_output_gate:
                return w  # native [Q|K|V] is already the C++ layout.
            rows = w.shape[0]
            # Derive the local q rows from the tensor itself so a global/local q_size mismatch
            # cannot silently skip the repack.
            local_q_rows = (rows - 2 * kv_size) // 2
            local_heads = local_q_rows // head_dim
            if local_q_rows <= 0 or local_heads * head_dim != local_q_rows:
                return w
            tail = w.shape[1:]
            q_gate, k_w, v_w = w.split([2 * local_q_rows, kv_size, kv_size], dim=0)
            qg = q_gate.reshape(local_heads, 2, head_dim, *tail)
            # C++ mha_qkv layout: [Q|K|V|Gate].
            return torch.cat(
                [qg[:, 0].reshape(local_q_rows, *tail), k_w, v_w, qg[:, 1].reshape(local_q_rows, *tail)], dim=0
            ).contiguous()

        # The masks insert the _DUMMY_TENSOR placeholders the C++ init skips, so every
        # family slot list stays index-aligned with the model layers.
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj", mask=full_mask)
        self.init_matmul_weights(layers, "mha_qkv", "self_attn.qkv_proj", mask=full_mask, row_processor=_repack_qkv)
        has_qkv_bias = any(
            (attn := getattr(layer, "self_attn", None)) is not None
            and getattr(getattr(attn, "qkv_proj", None), "bias", None) is not None
            for layer in layers
        )
        if has_qkv_bias:
            xlite_model.mha_qkv_bias = get_layer_weights(
                layers, "self_attn.qkv_proj.bias", mask=full_mask, post_processor=_repack_qkv
            )
            xlite_config.qkv_bias = True
        if xlite_config.qk_norm:
            xlite_model.mha_q_norm = get_layer_weights(
                layers, "self_attn.q_norm.weight", mask=full_mask, post_processor=self._gemma_norm
            )
            xlite_model.mha_k_norm = get_layer_weights(
                layers, "self_attn.k_norm.weight", mask=full_mask, post_processor=self._gemma_norm
            )

    def _bind_kda_layer_weights(self, layers: Sequence[nn.Module], kda_mask: Sequence[bool]) -> None:
        """KDA weights: in_proj_qkvz -> [q|k|v] + z, in_proj_ba -> b + a, plus conv/gate/out."""
        xlite_model = self.xlite_model
        hf_config = self.hf_text_config
        tp_size = self.xlite_config.def_tp_size
        k_dim = hf_config.linear_key_head_dim
        v_dim = hf_config.linear_value_head_dim
        n_k_local = max(1, hf_config.linear_num_key_heads // tp_size)
        n_v_local = max(1, hf_config.linear_num_value_heads // tp_size)
        kdl, vdl = n_k_local * k_dim, n_v_local * v_dim

        # Fused rows [q|k|v|z] and [b|a]; the row transform also slices their quant side tensors.
        self.init_matmul_weights(
            layers,
            "linear_in_proj_qkv",
            "linear_attn.in_proj_qkvz",
            mask=kda_mask,
            row_processor=_slice_rows(0, 2 * kdl + vdl),
        )
        self.init_matmul_weights(
            layers,
            "linear_in_proj_z",
            "linear_attn.in_proj_qkvz",
            mask=kda_mask,
            row_processor=_slice_rows(2 * kdl + vdl, vdl),
        )
        self.init_matmul_weights(
            layers,
            "linear_in_proj_b",
            "linear_attn.in_proj_ba",
            mask=kda_mask,
            row_processor=_slice_rows(0, n_v_local),
        )
        self.init_matmul_weights(
            layers,
            "linear_in_proj_a",
            "linear_attn.in_proj_ba",
            mask=kda_mask,
            row_processor=_slice_rows(n_v_local, n_v_local),
        )
        self.init_matmul_weights(layers, "linear_out_proj", "linear_attn.out_proj", mask=kda_mask)
        xlite_model.linear_conv1d = get_layer_weights(layers, "linear_attn.conv1d.weight", mask=kda_mask)
        xlite_model.linear_a_log = get_layer_weights(layers, "linear_attn.A_log", mask=kda_mask)
        xlite_model.linear_dt_bias = get_layer_weights(layers, "linear_attn.dt_bias", mask=kda_mask)
        # KDA gated RMSNorm is standard x*w already.
        xlite_model.linear_norm = get_layer_weights(layers, "linear_attn.norm.weight", mask=kda_mask)

    def _bind_shared_weights(self, layers: Sequence[nn.Module]) -> None:
        """Dense MLP slots plus MoE routed/shared expert weights (MoE layers only)."""
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        self._bind_dense_mlp(layers)
        if self.quantization:
            first_mha = next((w for w in xlite_model.mha_qkv if w.ndim >= 2), None)
            xlite_config.quant_attn_weight_nz = first_mha is not None and self.is_tensor_nz(first_mha)
            xlite_config.quant_attn_weight_transpose = True

        mlp_prefix = self._decoder_layer_mlp_module
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_model.gate_bias = get_layer_weights(
            layers, f"{mlp_prefix}.gate.e_score_correction_bias", post_processor=_to_fp32
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_expert.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_expert.down_proj")

        # FusedMoE stacks its local experts on dim 0.
        moe_layers = [layer for layer in layers if getattr(layer.mlp, "experts", None) is not None]
        if not moe_layers:
            return
        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)
        xlite_config.gate_captured = bool(xlite_model.gate) and self.is_tensor_nz(xlite_model.gate[0])
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])
        shared_experts = [
            layer.mlp.shared_expert for layer in moe_layers if getattr(layer.mlp, "shared_expert", None) is not None
        ]
        if shared_experts:
            xlite_model.se_gate = [layer.mlp.shared_expert_gate.weight for layer in moe_layers]
            xlite_config.n_shared_experts = 1
        if not self.quantization:
            return

        xlite_model.re_up_gate_scale_bias = get_layer_weights(layers, f"{re_prefix}.w13_scale_bias", **re_kwargs)
        xlite_model.re_down_scale_bias = get_layer_weights(layers, f"{re_prefix}.w2_scale_bias", **re_kwargs)
        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    # ---- attention metadata ----------------------------------------------------------------

    @staticmethod
    def _is_mamba_group(group: Any) -> bool:
        from vllm.v1.kv_cache_interface import MambaSpec

        return isinstance(group.kv_cache_spec, MambaSpec)

    def _main_attn_group(self, kv_cache_groups: Sequence[Any]) -> Any:
        return next((g for g in kv_cache_groups if not self._is_mamba_group(g)), kv_cache_groups[0])

    def _extra_attn_groups(self, kv_cache_groups: Sequence[Any]) -> Sequence[Any]:
        return [g for g in kv_cache_groups if self._is_mamba_group(g)]

    # ---- structured KV cache ---------------------------------------------------------------

    def _kv_cache_entry_count(self, layer: int) -> int:
        return 1

    def _make_layer_cache(self, layer: int, entries: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        cache = entries[0]
        return tuple(cache) if isinstance(cache, (list, tuple)) else (cache,)


class DeepseekV3XliteModel(Glm4MoeXliteModel):
    """xlite adapter for DeepseekV3 MoE architectures with MLA attention."""

    _supported_architectures = ["DeepseekV3ForCausalLM", "Glm4MoeLiteForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # MLA attention type
        xlite_config.attn_type = AttnMLA
        xlite_config.n_kv_heads = 1  # MLA uses latent cache
        xlite_config.head_dim = 0  # Ignored by MLA

        # MLA dimensions (override Llama's head_dim-based rope_head_dim)
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.qk_nope_head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.kv_lora_rank
        xlite_config.v_head_dim = hf_config.v_head_dim
        xlite_config.softmax_scale = (hf_config.qk_rope_head_dim + hf_config.qk_nope_head_dim) ** -0.5
        # correct softmax_scale for yarn-style RoPE if max_seq_len > original_max_position_embeddings
        rope_params: dict[str, int | float | str] = getattr(hf_config, "rope_parameters", {})
        original_max_len: int = rope_params.get("original_max_position_embeddings", hf_config.max_position_embeddings)  # type: ignore[assignment]
        if xlite_config.max_seq_len > original_max_len and "mscale" in rope_params and "factor" in rope_params:
            mscale: float = 1.0 + 0.1 * rope_params["mscale"] * math.log(rope_params["factor"])  # type: ignore[operator,arg-type]
            xlite_config.softmax_scale *= mscale**2

        # MoE configuration (from Glm4MoeXliteModel, adapted for DeepseekV3)
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()

        # MLA attention weights
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj")
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj")
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight")
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight")
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV")
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T")
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(wuv_lst[0])
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(wuk_lst[0])

        if not self.quantization:
            return

        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(wt_lst[0])
        self.xlite_config.quant_attn_weight_transpose = True
        with xlite_model.condition(lambda tensors: not self.all_tensors_zero(tensors)):
            xlite_model.mla_q_norm_bias = get_layer_weights(layers, "self_attn.q_a_layernorm.bias")
            xlite_model.mla_kv_norm_bias = get_layer_weights(layers, "self_attn.kv_a_layernorm.bias")

    def _precompute_rope_cache(self) -> torch.Tensor:
        # TODO: remove this method once `xlite` no longer requires complex64 format
        cossin_cache = super(StandardXliteModel, self)._precompute_rope_cache()
        if not self.rottary_embed:
            return cossin_cache.to(dtype=torch.complex64, device=self.device)
        # For MLA attention, we need to return the cache as a complex instead of separate cos/sin tensors
        if not (isinstance(cossin_cache, torch.Tensor) and cossin_cache.ndim == 2 and cossin_cache.shape[1] % 2 == 0):
            raise ValueError("RoPE cache must be a 2D tensor with even second dimension for MLA attention.")
        sep = cossin_cache.shape[-1] // 2
        return torch.complex(cossin_cache[:, :sep], cossin_cache[:, sep:]).to(dtype=torch.complex64, device=self.device)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        if self.xlite_config.kv_lora_rank == self.xlite_config.rope_head_dim:
            raise RuntimeError("xlite does not support `kv_lora_rank == rope_head_dim` for MLA attention.")
        n_layers = self.xlite_config.n_layers
        kv_nope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.kv_lora_rank, kv_caches))[:n_layers]
        k_rope_caches = list(filter(lambda x: x.shape[-1] == self.xlite_config.rope_head_dim, kv_caches))[:n_layers]
        if (n_nope := len(kv_nope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA KV nope caches ({n_nope} vs. {n_layers})")
        if (n_rope := len(k_rope_caches)) != n_layers:
            raise ValueError(f"xlite: failed to extract MLA K rope caches ({n_rope} vs. {n_layers})")
        return [(k_nope, k_rope) for k_nope, k_rope in zip(kv_nope_caches, k_rope_caches)]


class DeepseekV32XliteModel(DeepseekV3XliteModel):
    """xlite adapter for Deepseek-V3.2/GLM-5.x architectures with Deepseek sparse attention (DSA)."""

    _supported_architectures = ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"]

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.attn_type = AttnDSA
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        index_types: list[str] = list(map(lambda x: str(x).lower(), getattr(hf_config, "indexer_types", []) or []))
        xlite_config.index_full_mask = list(map(lambda x: x.startswith("full"), index_types))[: xlite_config.n_layers]

    def _build_model(self) -> None:
        super()._build_model()
        xlite_model = self.xlite_model
        layers, _ = self._get_layers_and_model_prefix()

        mask: list[bool] | None = getattr(self.xlite_config, "index_full_mask", None) or None
        self.init_matmul_weights(layers, "index_q_b", f"{(prefix := 'self_attn.indexer')}.wq_b", mask=mask)
        xlite_model.index_k_weights_proj = get_layer_weights(layers, f"{prefix}.wk_weights_proj.weight", mask=mask)
        # xlite backend expects index_k_norm and index_k_norm_bias to have dtype=torch.float32 or the model dtype
        xlite_model.index_k_norm = get_layer_weights(layers, f"{prefix}.k_norm.weight", mask=mask)
        xlite_model.index_k_norm_bias = get_layer_weights(layers, f"{prefix}.k_norm.bias", mask=mask)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        # For DSA, `kv_caches` is passed as [<index_k_cache (optional)>, kv_nope_cache, k_pe_cache, ..., <mtp_layer>].
        # Under MRV2, all `index_k_cache` are passed first; under V1, they are interleaved with the kv caches.
        # For xlite, the `kv_caches` is expected as [(kv_nope_cache, k_pe_cache, index_k_cache), ...] for each layer.
        # TODO: consider the compatibility with `enable_sparse_sfa_c8` and `enable_sparse_li_c8`
        xcfg = self.xlite_config
        if xcfg.index_head_dim == xcfg.kv_lora_rank or xcfg.index_head_dim == xcfg.rope_head_dim:
            raise RuntimeError("xlite does not support `index_head_dim == kv_lora_rank or rope_head_dim` for DSA.")
        index_mask: list[bool] = (getattr(xcfg, "index_full_mask", None) or [True] * xcfg.n_layers)[: xcfg.n_layers]

        mla_kv_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] != xcfg.index_head_dim, kv_caches))
        mla_caches: list[tuple[torch.Tensor, ...]] = super().extract_kv_cache(mla_kv_caches)
        index_caches: list[torch.Tensor] = list(filter(lambda x: x.shape[-1] == xcfg.index_head_dim, kv_caches))
        if (n_index_layers := sum(index_mask)) != len(index_caches := index_caches[:n_index_layers]):
            raise ValueError(f"xlite: index_k_cache count mismatch ({len(index_caches)} vs. {n_index_layers})")
        if wrong_dims := set(cache.shape[-1] for cache in index_caches) - set([xcfg.index_head_dim]):
            raise ValueError(f"xlite: index_k_cache dimension mismatch (wrong dims: {', '.join(map(str, wrong_dims))})")
        for i, mask in enumerate(index_mask):
            if not mask:
                index_caches.insert(i, _DUMMY_TENSOR)
        return [(mla_cache[0], mla_cache[1], i_cache) for mla_cache, i_cache in zip(mla_caches, index_caches)]


class Glm5NextXliteModel(HybridXliteModel, DeepseekV32XliteModel):
    """xlite adapter for GLM-5.3-Flash (glm5_next): hybrid KDA linear attention + NoPE sparse MLA + MHC + MoE.

    Reuses the DeepseekV32 DSA plumbing (SFA metadata, indexer cache slots) but rebuilds the
    config/weight binding for the two attention families (KDA ``linear_attention`` with fused
    ``in_proj_qkvbfg_a``, sparse MLA with per-layer kpool indexer) and the MHC residual streams.

    Metadata is assembled into :class:`AttnMetaV2` from the runner's raw per-family block tables;
    KV caches are consumed structurally: a sparse-MLA layer owns ``(main, indexer, tail)`` runner
    entries and a KDA layer owns one ``(conv, ssm)`` entry.
    """

    from vllm_ascend.attention.indexer_kpool import (  # noqa: F401  (re-exported for tests)
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )

    _supported_architectures = ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]

    def _build_model_config(self) -> None:
        # Skip the DeepseekV3/V32 chain: GLM5 has no standard qkv_proj, its indexer fields fold
        # differently, and KDA/MLA dispatch is explicit. The Standard base plus the GLM5-specific
        # fields below cover the full config surface.
        if getattr(self.vllm_config.parallel_config, "prefill_context_parallel_size", 1) > 1:
            raise NotImplementedError("xlite glm5: prefill context parallelism is not supported.")
        # Glm5NextConfig declares head_dim=None (MLA latent layout, no per-head dim), but the
        # Standard base does `head_dim**-0.5` unconditionally. Substitute a placeholder for the
        # base's arithmetic (overwritten below) and always restore the original afterwards.
        saved_head_dim = getattr(self.hf_text_config, "head_dim", None)
        try:
            if not saved_head_dim:  # None or 0 (MLA latent layout has no per-head dim)
                self.hf_text_config.head_dim = 1
            StandardXliteModel._build_model_config(self)
        finally:
            if not saved_head_dim:
                self.hf_text_config.head_dim = saved_head_dim
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        # Attention type & MLA latent dims (NoPE sparse MLA: rope only on the K tail).
        # AttnHybridKda: dedicated C++ dispatch (per-layer KDA-linear vs sparse-MLA via kda_layers).
        xlite_config.attn_type = AttnHybridKda
        xlite_config.n_kv_heads = 1
        xlite_config.head_dim = 0
        xlite_config.rope_head_dim = hf_config.qk_rope_head_dim
        xlite_config.nope_head_dim = hf_config.qk_nope_head_dim
        xlite_config.v_head_dim = hf_config.v_head_dim
        xlite_config.q_lora_rank = hf_config.q_lora_rank
        xlite_config.kv_lora_rank = hf_config.kv_lora_rank
        xlite_config.softmax_scale = (hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim) ** -0.5

        # MHC residual mixing (transformers 5.16.1~5.19 字段即 hc_mult/hc_sinkhorn_iters,config.json 同名).
        xlite_config.hc_mult = hf_config.hc_mult
        xlite_config.hc_sinkhorn_iters = hf_config.hc_sinkhorn_iters
        xlite_config.hc_eps = hf_config.hc_eps

        # KDA linear-attention dims (q/k/v share the same head layout).
        self._set_kda_config(
            hf_config.linear_num_heads,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            hf_config.linear_head_dim,
            hf_config.linear_conv_kernel_dim,
        )
        # KDA safe_gate lower bound (C++ gateLowerBound; native: kda.py kda_lower_bound).
        xlite_config.gate_lower_bound = float(hf_config.linear_lower_bound)

        # kpool indexer (sparse MLA layers): kpool>1 enables the compress+spill path.
        compress_ratio = int(hf_config.index_kpool)
        xlite_config.index_kpool = compress_ratio
        xlite_config.index_kpool_compress = compress_ratio > 1
        xlite_config.index_head_dim = hf_config.index_head_dim
        xlite_config.index_n_heads = hf_config.index_n_heads
        xlite_config.index_topk = hf_config.index_topk
        xlite_config.index_softmax_scale = hf_config.index_head_dim**-0.5
        xlite_config.index_rope_interleaved = getattr(hf_config, "indexer_rope_interleave", False)
        # All GLM5 sparse layers use the full (non-sliding) indexer mask; per-layer KDA-vs-MLA
        # attention dispatch is carried by the C++ model init (the masks below stay python-side
        # for weight mapping only).
        xlite_config.index_full_mask = [True] * xlite_config.n_layers

        # block_sizes is the family vector [main kernel page, pooled entries per main page, tail
        # ring capacity] and compress_ratios is per-layer (the C++ derives the indexer pool/tail
        # slots from these). XModel copies this config at construction, so the values must be
        # pinned here — patching after C++ init has no effect. The shared hook also pins
        # block_size and rounds max_seq_len up to whole manager pages.
        from vllm_ascend.attention.sfa_v1 import AscendSFABackend
        from vllm_ascend.models.glm5next.kv_cache import get_kpool_tail_ring_capacity

        main_block = self._pin_cache_geometry([AscendSFABackend])
        xlite_config.block_sizes = [
            main_block,
            main_block // compress_ratio,
            get_kpool_tail_ring_capacity(self.vllm_config, compress_ratio),
        ]
        xlite_config.compress_ratios = [compress_ratio] * xlite_config.n_layers

        # MoE (W8A8 routed/shared experts, BF16 gate; dense layers use standard MLP slots).
        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = getattr(hf_config, "routed_scaling_factor", 1.0)
        xlite_config.swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0) or 0.0
        xlite_config.gate_captured = False  # GLM5 router gate is BF16 (never NZ-captured)

    def _init_hybrid_layers(self, layers: Sequence[nn.Module]) -> list[int]:
        # Per-layer KDA-vs-MLA dispatch from layer_types. Pad a short layer_types (e.g. when the
        # MTP layer is not listed) with sparse-MLA, matching the tail-layer layout. Python-side
        # only: drives the weight masks below and the KV cache family assembly.
        layer_types: list[str] = list(getattr(self.hf_text_config, "layer_types", []) or [])
        if len(layer_types) < len(layers):
            layer_types += ["deepseek_sparse_attention"] * (len(layers) - len(layer_types))
        return [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    def _bind_mla_weights(self, layers: Sequence[nn.Module], mask: Sequence[bool]) -> None:
        """Bind the sparse-MLA latent projections, q/kv norms and per-head W_UV / W_UK_T slots."""
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj", mask=mask)
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj", mask=mask)
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight", mask=mask)
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight", mask=mask)
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV", mask=mask)
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T", mask=mask)
        # Masked lists pad missing layers with the 1-D dummy, so skip placeholders when probing NZ.
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(
            next((w for w in wuv_lst if w.ndim >= 2), wuv_lst[0])
        )
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(
            next((w for w in wuk_lst if w.ndim >= 2), wuk_lst[0])
        )

    def _bind_full_layer_weights(self, layers: Sequence[nn.Module], full_mask: Sequence[bool]) -> None:
        """Sparse-MLA + kpool indexer slots (non-KDA layers, i.e. ``full_mask``)."""
        self._bind_mla_weights(layers, full_mask)
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj", mask=full_mask)
        self._bind_indexer_weights(layers, full_mask)

    def _bind_kda_layer_weights(self, layers: Sequence[nn.Module], kda_mask: Sequence[bool]) -> None:
        """KDA (BF16) fused projection shards, conv/gate params and output projection."""
        xlite_model = self.xlite_model
        hf_config = self.hf_text_config
        tp_size = self.xlite_config.def_tp_size
        # Fused in_proj_qkvbfg_a row shards: [q, k, v (proj each), b (heads), f_a, g_a (head_dim,
        # replicated)]. The 2D row slice must no-op on the 1D _DUMMY_TENSOR for masked-out layers.
        proj = hf_config.linear_head_dim * hf_config.linear_num_heads
        lp = proj // tp_size  # local projection (q/k/v per rank)
        lh = hf_config.linear_num_heads // tp_size  # local heads (b per rank)
        hd = hf_config.linear_head_dim
        slices = {
            "glm5_q_proj": (0, lp),
            "glm5_k_proj": (lp, lp),
            "glm5_v_proj": (2 * lp, lp),
            "glm5_b_proj": (3 * lp, lh),
            "glm5_f_a_proj": (3 * lp + lh, hd),
            "glm5_g_a_proj": (3 * lp + lh + hd, hd),
        }
        for slot, (start, length) in slices.items():
            setattr(
                xlite_model,
                slot,
                get_layer_weights(
                    layers,
                    "self_attn.in_proj_qkvbfg_a.weight",
                    mask=kda_mask,
                    post_processor=_slice_rows(start, length),
                ),
            )
        self.init_matmul_weights(layers, "glm5_f_b_proj", "self_attn.f_b_proj", mask=kda_mask)
        self.init_matmul_weights(layers, "glm5_g_b_proj", "self_attn.g_b_proj", mask=kda_mask)
        self.init_matmul_weights(layers, "glm5_wo", "self_attn.o_proj", mask=kda_mask)
        xlite_model.glm5_q_conv1d = get_layer_weights(layers, "self_attn.q_conv1d.weight", mask=kda_mask)
        xlite_model.glm5_k_conv1d = get_layer_weights(layers, "self_attn.k_conv1d.weight", mask=kda_mask)
        xlite_model.glm5_v_conv1d = get_layer_weights(layers, "self_attn.v_conv1d.weight", mask=kda_mask)
        # Conv1d weights are stored fp32; the C++ kernel requires weight/activation dtype match,
        # so mirror native's one-time cast to the activation dtype (kda.py _merged_conv_weight).
        # NOTE: pybind def_readwrite returns a COPY of the C++ vector, so write back via setattr.
        for conv_name in ("glm5_q_conv1d", "glm5_k_conv1d", "glm5_v_conv1d"):
            conv_w = getattr(xlite_model, conv_name)
            if conv_w and conv_w[0].dtype != self.dtype:
                setattr(xlite_model, conv_name, [w.to(self.dtype) for w in conv_w])
        # rc2 stores A_log as [1, 1, local_heads, 1]; the C++ reference consumes 1-D [local_heads].
        xlite_model.glm5_a_log = get_layer_weights(
            layers, "self_attn.A_log", mask=kda_mask, post_processor=lambda w: w.reshape(-1) if w.ndim > 1 else w
        )
        xlite_model.glm5_dt_bias = get_layer_weights(layers, "self_attn.dt_bias", mask=kda_mask)
        xlite_model.glm5_o_norm = get_layer_weights(layers, "self_attn.o_norm.weight", mask=kda_mask)

    def _bind_indexer_weights(self, layers: Sequence[nn.Module], mla_mask: Sequence[bool]) -> None:
        """kpool indexer weights (MLA layers), cast to fp32 as native does after loading."""
        xlite_model = self.xlite_model
        idx_hd = self.hf_text_config.index_head_dim
        self.init_matmul_weights(layers, "glm5_idx_wq_b", "self_attn.indexer.wq_b", mask=mla_mask)
        # wk_weights_proj is fused [wk (idx_hd rows); weights_proj] — split per the reference.
        xlite_model.glm5_idx_wk = get_layer_weights(
            layers,
            "self_attn.indexer.wk_weights_proj.weight",
            mask=mla_mask,
            post_processor=lambda w: _to_fp32(w[:idx_hd, :].contiguous() if w.ndim >= 2 else w),
        )
        xlite_model.glm5_idx_weights_proj = get_layer_weights(
            layers,
            "self_attn.indexer.wk_weights_proj.weight",
            mask=mla_mask,
            post_processor=lambda w: _to_fp32(w[idx_hd:, :].t().contiguous() if w.ndim >= 2 else w),
        )
        xlite_model.glm5_idx_k_norm = get_layer_weights(
            layers, "self_attn.indexer.k_norm.weight", mask=mla_mask, post_processor=_to_fp32
        )
        xlite_model.glm5_idx_k_norm_bias = get_layer_weights(
            layers, "self_attn.indexer.k_norm.bias", mask=mla_mask, post_processor=_to_fp32
        )
        xlite_model.glm5_idx_kpool_gate = get_layer_weights(
            layers, "self_attn.indexer.index_kpool_compress_gate", mask=mla_mask, post_processor=_to_fp32
        )
        # native validates the APE as fp32 [kpool, head_dim]; cast defensively.
        xlite_model.glm5_idx_kpool_ape = get_layer_weights(
            layers, "self_attn.indexer.index_kpool_compress_ape", mask=mla_mask, post_processor=_to_fp32
        )

    def _bind_mhc_weights(self, layers: Sequence[nn.Module]) -> None:
        """MHC residual-mixing params (fp32, every layer)."""
        for slot in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            setattr(self.xlite_model, slot, get_layer_weights(layers, slot, post_processor=_to_fp32))

    def _bind_moe_weights(self, layers: Sequence[nn.Module]) -> None:
        """Dense MLP slots and routed/shared expert weights."""
        xlite_model = self.xlite_model
        mlp_prefix = self._decoder_layer_mlp_module
        self._bind_dense_mlp(layers)
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_model.gate_bias = get_layer_weights(
            layers, f"{mlp_prefix}.gate.e_score_correction_bias", post_processor=_to_fp32
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")
        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)

    def _bind_quant_scales(self, layers: Sequence[nn.Module]) -> None:
        """Quant deq-scale tables: sparse-MLA attention and routed experts."""
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        # Attention quant applies to the sparse-MLA latent projections (KDA stays BF16);
        # sample NZ-ness from the first real (non-placeholder) entry.
        xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(
            next((w for w in wt_lst if w.ndim >= 2), wt_lst[0])
        )
        xlite_config.quant_attn_weight_transpose = True
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])
        re_prefix = f"{self._decoder_layer_mlp_module}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate_scale_bias = get_layer_weights(layers, f"{re_prefix}.w13_scale_bias", **re_kwargs)
        xlite_model.re_down_scale_bias = get_layer_weights(layers, f"{re_prefix}.w2_scale_bias", **re_kwargs)
        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def _bind_shared_weights(self, layers: Sequence[nn.Module]) -> None:
        """MHC residual mixing, dense MLP/MoE and quant scales."""
        self._bind_mhc_weights(layers)
        self._bind_moe_weights(layers)
        if self.quantization:
            self._bind_quant_scales(layers)

    # ---- attention metadata ----------------------------------------------------------------

    @staticmethod
    def _glm5_cache_layout(kv_cache_groups: Sequence[Any]) -> Any:
        from vllm_ascend.models.glm5next.cache_config import _get_glm5_next_cache_layout

        layout = _get_glm5_next_cache_layout(list(kv_cache_groups))
        if layout is None:
            raise ValueError("xlite glm5: unrecognized KV cache group layout")
        return layout

    def _main_attn_group(self, kv_cache_groups: Sequence[Any]) -> Any:
        return self._glm5_cache_layout(kv_cache_groups).full_group

    def _extra_attn_groups(self, kv_cache_groups: Sequence[Any]) -> Sequence[Any]:
        layout = self._glm5_cache_layout(kv_cache_groups)
        groups = list(layout.mamba_groups)
        # Match the current GVirt runtime: the tail-ring family exists only for
        # kpool compression (indexKpool > 1).
        if self.xlite_config.index_kpool > 1:
            groups.append(layout.tail_group)
        return groups

    # ---- structured KV cache ---------------------------------------------------------------

    @classmethod
    def _cache_component(cls, entry: Any, width: int, family: str, layer: int) -> torch.Tensor:
        """Select the ``width``-lane tensor from a possibly multi-tensor cache entry."""
        entry = cls._unwrap_cache_entry(entry)
        if isinstance(entry, torch.Tensor):
            components = [entry]
        else:
            components = list(entry) if isinstance(entry, (list, tuple)) else []
        matches = [c for c in components if isinstance(c, torch.Tensor) and c.shape[-1] == width]
        if len(matches) != 1:
            raise ValueError(
                f"xlite glm5: layer {layer} {family} cache entry has {len(matches)} tensors with last dim {width}"
            )
        return matches[0]

    def _as_block_major(self, view: Any, width: int, family: str, layer: int) -> torch.Tensor:
        """Normalize a runner MLA/indexer view to ``[blocks, states, width]``."""
        view = self._unwrap_cache_entry(view)
        if view.dim() == 4:
            # MLA caches carry one latent head at dim 1 in some backends and dim 2 in others.
            if view.shape[1] == 1:
                view = view.squeeze(1)
            elif view.shape[2] == 1:
                view = view.squeeze(2)
        if view.dim() != 3 or view.shape[-1] != width:
            raise ValueError(
                f"xlite glm5: layer {layer} {family} cache view {tuple(view.shape)} does not collapse to "
                f"[blocks, states, {width}]"
            )
        return view

    def _validate_kda_pool(
        self, conv: torch.Tensor, ssm: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate the KDA pool pair against the native layer's declared state geometry."""
        cfg = self.xlite_config
        layers, _ = self._get_layers_and_model_prefix()
        attn = layers[layer].self_attn
        get_state_shape = getattr(attn, "get_state_shape", None)
        if callable(get_state_shape):
            # Native shapes already include the speculative lookahead (`num_spec`).
            conv_shape, ssm_shape = get_state_shape()[:2]
            conv = self._expect_cache_shape(conv, tuple(conv_shape), "conv pool", layer)
            ssm = self._expect_cache_shape(ssm, tuple(ssm_shape), "ssm pool", layer)
        else:
            heads = cfg.linear_num_k_heads // cfg.def_tp_size
            dim = cfg.linear_key_head_dim
            conv = self._expect_cache_shape(
                conv, (3 * (cfg.linear_conv_kernel_dim - 1), heads * dim), "conv pool", layer
            )
            ssm = self._expect_cache_shape(ssm, (heads, dim, dim), "ssm pool", layer)
        if conv.shape[0] != ssm.shape[0] or conv.shape[0] == 0:
            raise ValueError(f"xlite glm5: layer {layer} conv/ssm slot counts differ or are empty")
        get_state_dtype = getattr(attn, "get_state_dtype", None)
        if callable(get_state_dtype):
            conv_dtype, _ = get_state_dtype()[:2]
            if conv.dtype != conv_dtype:
                raise ValueError(f"xlite glm5: layer {layer} conv pool must be {conv_dtype}, got {conv.dtype}")
        if ssm.dtype != torch.float32:
            raise ValueError(f"xlite glm5: layer {layer} ssm pool must be fp32, got {ssm.dtype}")
        return conv, ssm

    @classmethod
    def _cache_components(cls, entry: Any) -> list[torch.Tensor]:
        entry = cls._unwrap_cache_entry(entry)
        if isinstance(entry, torch.Tensor):
            return [entry]
        return [c for c in entry if isinstance(c, torch.Tensor)] if isinstance(entry, (list, tuple)) else []

    def _ensure_tail_rings(self) -> list[torch.Tensor]:
        if (rings := getattr(self, "_tail_rings", None)) is not None:
            return rings
        cfg = self.xlite_config
        if cfg.block_sizes[2] != cfg.index_kpool:
            raise ValueError(
                f"xlite glm5: tail ring capacity {cfg.block_sizes[2]} != indexKpool {cfg.index_kpool} "
                "(spec-decode lookahead is not supported here)."
            )
        self._tail_rings = [
            torch.zeros(
                cfg.max_batch_size,
                2,
                cfg.index_kpool,
                cfg.index_head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(len(self.full_attn_layers))
        ]
        return self._tail_rings

    def _mla_cache(self, layer: int, main: Any, indexer: Any, tail: Any) -> tuple[torch.Tensor, ...]:
        xcfg = self.xlite_config
        components = self._cache_components(main)
        k_cache = next((c for c in components if c.shape[-1] == xcfg.kv_lora_rank), None)
        if k_cache is None:
            raise ValueError(f"xlite glm5: layer {layer} main cache has no {xcfg.kv_lora_rank}-wide view.")
        pe_cache = next(
            (c for c in components if c is not k_cache and c.shape[-1] in (xcfg.rope_head_dim, 0)),
            k_cache[..., :0],
        )
        indexer = self._cache_component(indexer, xcfg.index_head_dim, "indexer kpool", layer)
        tail_ring = self._ensure_tail_rings()[self.full_attn_layers.index(layer)]
        return k_cache, pe_cache, indexer, tail_ring

    def _kv_cache_entry_count(self, layer: int) -> int:
        # A sparse-MLA layer owns [main, indexer, tail]; a KDA layer one [conv, ssm] entry.
        return 3 if layer in self.full_attn_layers else 1

    def _make_layer_cache(self, layer: int, entries: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        if layer in self.full_attn_layers:
            return self._mla_cache(layer, *entries)
        conv, ssm = entries[0]
        return self._validate_kda_pool(conv, ssm, layer)


class MiniMaxM2XliteModel(StandardXliteModel):
    """xlite adapter for MiniMax M2 architectures."""

    _supported_architectures = ["MiniMaxM2ForCausalLM"]
    _decoder_layer_mlp_module = "block_sparse_moe"

    def _build_model_config(self) -> None:
        super()._build_model_config()
        xlite_config, hf_config = self.xlite_config, self.hf_text_config

        xlite_config.rope_head_dim = hf_config.rotary_dim
        xlite_config.n_dense_layers = 0
        xlite_config.n_routed_experts = hf_config.num_local_experts
        xlite_config.n_shared_experts = 0
        xlite_config.n_act_experts = hf_config.num_experts_per_tok
        xlite_config.moe_intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.qk_norm_full = True
        xlite_config.scoring_func = ScoringFuncSigmoid


def get_adapter_xlite_model(npu_runner: XliteModelRunner, vllm_config: VllmConfig) -> XliteModelBase:
    """Look up and initialize the appropriate xlite model adapter based on the architecture specified in vLLM config and
    the runnable model.

    Args:
        npu_runner (XliteModelRunner): The NPU model runner instance.
        vllm_config (VllmConfig): Runtime configuration for model execution.

    Raises:
        ValueError: If the model architecture is not supported by xlite.

    Returns:
        XliteModelBase: An initialized xlite model adapter ready for inference.
    """
    architecture = vllm_config.model_config.architectures[0]
    if not (strategy_class := _architecture_strategy_map.get(architecture)):
        raise ValueError(f"{architecture} not supported!")
    return strategy_class(npu_runner, vllm_config)


class XliteWrapper:
    """A graph-based wrapper that dispatches between xlite and runnable paths.

    Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention state:
    batches within the preallocated budget (:attr:`max_tokens`, sized by ``full_mode``) run on xlite, others fall
    back to the native runnable.
    """

    def __init__(self, npu_runner: XliteModelRunner, vllm_config: VllmConfig, device: torch.device) -> None:
        """Initialize xlite runtime, model tensors, and hidden-state workspace.

        Args:
            npu_runner (XliteModelRunner): The native NPU model runner.
            vllm_config (VllmConfig): Runtime configuration for execution.
            device (torch.device): The device to initialize the xlite model on.

        Raises:
            ValueError: If xlite runtime tensor-pool initialization fails.
        """
        self.npu_runner: XliteModelRunner = npu_runner
        self.npu_runnable: nn.Module = npu_runner.fallback_model
        self.device = device
        # `full_mode` only sizes the token budget (see the comment in `StandardXliteModel._build_config`).
        self.full_mode: bool = get_ascend_config().xlite_graph_config.full_mode
        self.data_parallel_size = vllm_config.parallel_config.data_parallel_size
        # The runner generation decides the output-shape contract; see `XliteWrapper.__call__`.
        self.use_v2_model_runner: bool = vllm_config.use_v2_model_runner
        self._is_dummy_batch: bool = False

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
        # V2 hybrid 模型缺 per-group common metadata 生产者(基线只接了 default state),安装桥接。
        if self.use_v2_model_runner and isinstance(self.adapter_xlite_model, HybridXliteModel):
            install_hybrid_common_metadata_bridge()
        self.xlite_model = self.adapter_xlite_model.xlite_model  # xlite C++ model instance
        self.freq_cis = self.adapter_xlite_model.rope_cossin_cache
        xlite_config = self.adapter_xlite_model.xlite_config
        self.xlite_rt = Runtime(
            devid=device.index,
            size=0,
            rank=torch.distributed.get_rank(),
            tp_size=xlite_config.def_tp_size,
            dp_size=xlite_config.def_dp_size,
            moe_tp_size=xlite_config.moe_tp_size,
            moe_ep_size=xlite_config.moe_ep_size,
        )
        # The xlite token budget: batches with more tokens (max across DP ranks) are routed to the native runnable.
        self.max_tokens = xlite_config.max_m
        self.hidden_states = torch.empty(
            self.max_tokens, xlite_config.hidden_size, device=self.device, dtype=self.adapter_xlite_model.dtype
        )
        # Persistent one-token batch for DP dummy runs (`__call__`): constant, allocated once.
        self._dummy_attn_metadata = self._init_dummy_metadata(xlite_config)

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if torch.distributed.get_rank() == 0:
            logger.info("xlite runtime pool size: %s MB", rt_pool_size)
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB")

    def __del__(self) -> None:
        self.npu_runner = None  # type: ignore[assignment]
        self.npu_runnable = None  # type: ignore[assignment]
        # __init__ may have failed before the adapter was created; go through __dict__ so a
        # half-constructed wrapper does not hit __getattr__ and mask the original error.
        if (adapter := self.__dict__.get("adapter_xlite_model")) is not None:
            adapter.npu_runner = None  # type: ignore[assignment]
            adapter.xlite_model = None  # type: ignore[assignment]

    def __getattr__(self, key: str) -> Any:
        """Proxy unknown attributes to the wrapped runnable model.

        Args:
            key (str): The attribute name requested by the caller.

        Raises:
            AttributeError: If neither wrapper nor runnable has the attribute.

        Returns:
            Any: Attribute value resolved from the runnable.
        """
        try:
            return getattr(self.npu_runnable, key)
        except Exception as e:  # runnable may raise various exceptions
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}") from e

    def register_kv_caches(self, kv_caches: list[Any] | dict[str, Any]) -> None:
        """Register KV cache references used by xlite runtime.

        V2 hands the structured per-layer views, or the runner's ``name -> view`` mapping for
        adapters that resolve ownership by layer id; legacy adapters keep receiving a flat list.

        Args:
            kv_caches: Runtime KV cache views, one entry per layer, a flat list, or the V2
                runner's name mapping.
        """
        if self.structured_kv_cache and not self.use_v2_model_runner:
            raise NotImplementedError("xlite: structured KV cache adapters require the V2 model runner.")
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)

    @property
    def structured_kv_cache(self) -> bool:
        """Whether the active adapter consumes structured per-layer cache views."""
        return bool(getattr(self.adapter_xlite_model, "_structured_kv_cache", False))

    @contextmanager
    def dummy_batch(self) -> Generator[None, None, None]:
        """Mark the enclosed run as a DP dummy batch.

        Idle DP ranks run ``NPUWorker.execute_dummy_batch`` through the xlite forward to keep the DP collectives
        aligned. xlite only requires such a rank to contribute a minimum of one token for one request, so while
        the flag is set the wrapper clamps every request-derived quantity to exactly that. The DP-agreed output
        buffer size is unaffected (see `XliteWrapper.__call__`).
        """
        self._is_dummy_batch = True
        try:
            yield
        finally:
            self._is_dummy_batch = False

    def _init_dummy_metadata(self, xlite_config: Any) -> AttnMetaV2:
        """Build the constant one-token batch used for DP dummy runs (`dummy_batch()`).

        Idle DP ranks run ``NPUWorker.execute_dummy_batch`` through the xlite forward to keep the DP
        collectives aligned; xlite only requires such a rank to contribute one token for one request.
        Its KV write lands on block 0 / slot 0 鈥?not provably free; xlite's kernels have no
        negative-slot skip yet.
        """
        dummy = AttnMetaV2()
        dummy.lens_cpu = [1]
        dummy.cached_lens_cpu = [0]
        dummy.lens = torch.ones(1, dtype=torch.int32, device=self.device)
        dummy.cached_lens = torch.zeros(1, dtype=torch.int32, device=self.device)
        dummy.query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        pos_dim = 3 if xlite_config.mrope_section else 1
        dummy.positions = torch.zeros((pos_dim, 1), dtype=torch.int64, device=self.device)
        dummy.slot_mapping = [torch.zeros(1, dtype=torch.int32, device=self.device)]
        dummy.block_tables = [torch.zeros((1, 1), dtype=torch.int32, device=self.device)]
        return dummy

    def _common_attn_metadata_by_group(self) -> dict[int, AscendCommonAttentionMetadata]:
        """Return the per-group common attention metadata retained by the runner.

        V2 stores one entry per KV cache group under ``model_state._decoder_replay_common_metadata``;
        V1 keeps a single group-0 object on the runner.
        """
        if (model_state := getattr(self.npu_runner, "model_state", None)) is not None:
            # model_state 未初始化 decoder replay(如模型无 decoder_replay_layers 时基座
            # _initialize_decoder_replay 提前 return)时返回空表,而非 AttributeError
            return getattr(model_state, "_decoder_replay_common_metadata", None) or {}
        common = getattr(self.npu_runner, "_decoder_replay_common_metadata", None)
        return {0: common} if common is not None else {}

    def _kv_cache_groups(self) -> list[Any]:
        config = getattr(self.npu_runner, "kv_cache_config", None)
        return list(config.kv_cache_groups) if config is not None else []

    def _common_attn_metadata(self) -> AscendCommonAttentionMetadata:
        """Return the main full-attention group's common attention metadata.

        Homogeneous models keep group 0 exactly as before. Hybrid adapters (KDA may own group 0)
        select the group that owns the full-attention KV cache.
        """
        select = getattr(self.adapter_xlite_model, "select_main_common_group", None)
        if select is None:
            if (model_state := getattr(self.npu_runner, "model_state", None)) is not None:
                common_by_group = getattr(model_state, "_decoder_replay_common_metadata", None) or {}
                return common_by_group[0]
            return self.npu_runner._decoder_replay_common_metadata
        return select(self._common_attn_metadata_by_group(), self._kv_cache_groups())

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> XliteForwardResult:
        """Run one forward step through xlite graph or fallback runnable path.

        Args:
            input_ids (torch.Tensor | None): Token IDs for current step.

            positions (torch.Tensor): Position IDs used by attention.
            intermediate_tensors (IntermediateTensors | None): Optional intermediate tensors from pipeline stages.
            inputs_embeds (torch.Tensor | None): Optional external input embeddings (e.g. multimodal/deepstack
                scenarios).
            **model_kwargs (Any): Additional keyword arguments for the runnable.

        Returns:
            XliteForwardResult: Forward outputs from xlite graph or the original runnable implementation.
        """
        forward_context = get_forward_context()
        common = self._common_attn_metadata()
        # Route by token count, max across DP ranks (graph modes hold the bucket-padded count 鈥?safe:
        # xlite only requires DP-uniform output rows never smaller than the rank's own token count).
        num_tokens: int
        if (num_tokens := _EXTRA_CTX.max_tokens_across_dp) is None:
            if (dp_metadata := forward_context.dp_metadata) is not None:
                num_tokens = dp_metadata.num_tokens_across_dp_cpu.max().item()
            else:
                num_tokens = common.num_actual_tokens
        if num_tokens > self.max_tokens:
            if self.full_mode:
                logger.warning_once(
                    "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                    num_tokens,
                    self.max_tokens,
                    "https://atomgit.com/openeuler/GVirt",
                )
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        # `input_ids` / `inputs_embeds` length = padded token count = the output rows consumers expect.
        if input_ids is not None:
            num_padded_input_tokens = input_ids.size(0)
        elif inputs_embeds is not None:
            num_padded_input_tokens = inputs_embeds.size(0)
        else:  # defensive: the model always receives at least one of the two
            num_padded_input_tokens = common.num_actual_tokens

        num_actual_tokens = common.num_actual_tokens
        # Rows handed to xlite: the true token count, or exactly one token for a DP dummy batch.
        num_forward_tokens = 1 if self._is_dummy_batch else num_actual_tokens

        if self._is_dummy_batch:
            xlite_attn_metadata = self._dummy_attn_metadata
        else:
            # V2's common is request-exact; V1 execution modes append padding requests whose qsl ends
            # exceed `num_actual_tokens` (real ends never do) 鈥?drop them there.
            if self.use_v2_model_runner:
                num_reqs = common.num_reqs
            else:
                qsl_cpu = common.query_start_loc_cpu
                num_reqs = max(int((qsl_cpu[1 : common.num_reqs + 1] <= num_actual_tokens).sum()), 1)
            # V1 async spec decode keeps only the optimistic `_seq_lens_cpu` upper bound here.
            seq_lens_cpu = common.seq_lens_cpu if common.seq_lens_cpu is not None else common._seq_lens_cpu
            query_lens = torch.diff(common.query_start_loc_cpu[: num_reqs + 1])
            # `lens` / `cached_lens` are the only derived tensors (no backend ships them); everything else
            # passes through as views of the runner's buffers 鈥?safe because xlite brackets its forward
            # with stream events (`EventWaitCurrStream` before, `EventRecordCurrStream` after), so both
            # the metadata-producing ops and the next step's in-place buffer rewrites are stream-ordered
            # against the forward's reads.
            query_start_loc = common.query_start_loc[: num_reqs + 1]
            lens = query_start_loc[1:] - query_start_loc[:-1]
            cached_lens = torch.clamp(common.seq_lens[:num_reqs] - lens, min=0)
            xlite_attn_metadata = AttnMetaV2()
            xlite_attn_metadata.lens_cpu = query_lens.tolist()
            xlite_attn_metadata.cached_lens_cpu = torch.clamp(seq_lens_cpu[:num_reqs] - query_lens, min=0).tolist()
            xlite_attn_metadata.lens = lens
            xlite_attn_metadata.cached_lens = cached_lens
            xlite_attn_metadata.query_start_loc = query_start_loc
            xlite_attn_metadata.positions = positions
            xlite_attn_metadata.slot_mapping = [common.slot_mapping]
            xlite_attn_metadata.block_tables = [common.block_table_tensor]

        # Hybrid adapters append their mamba/tail family tables to the common main-KV slot.
        augment = getattr(self.adapter_xlite_model, "augment_attn_meta", None)
        if augment is not None:
            augment(
                xlite_attn_metadata,
                self._common_attn_metadata_by_group(),
                self._kv_cache_groups(),
                self._is_dummy_batch,
            )

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        input_ids = cast(torch.Tensor, input_ids)
        curr_stream = torch.npu.current_stream().npu_stream  # type: ignore[attr-defined]
        if curr_stream == 0:
            logger.warning_once(
                "xlite: current stream handle is 0; xlite will fall back to blocking stream synchronization."
            )
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            self.xlite_model.forward_v2(
                rt=self.xlite_rt,
                input=input_ids[:num_forward_tokens],
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=[self.freq_cis],
                output=h,
                curr_stream=curr_stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_forward_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            # Unlike `forward_v2` (whose binding takes `std::vector<at::Tensor>`), the
            # `forward_with_inputs_embeds_v2` binding takes a single `at::Tensor` for `freqs_cis`.
            self.xlite_model.forward_with_inputs_embeds_v2(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=xlite_attn_metadata,
                kv_cache=self.kv_caches,
                freqs_cis=self.freq_cis,
                output=h,
                curr_stream=curr_stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if deepstack_input_embeds and callable(
                _clear_stack := getattr(self.npu_runnable, "_clear_deepstack_input_embeds", None)
            ):
                _clear_stack(inputs_embeds.size(0))
        if self.use_v2_model_runner:
            # V2 hands the model padded input views (`num_tokens_after_padding` rows) and its consumers require
            # exactly that many output rows; e.g. the MTP speculator's `hidden_states[:num_tokens_padded].copy_`
            # cannot broadcast a different first dimension. This equals `num_actual_tokens` in the eager modes
            # xlite runs under V2, and the capture bucket under (explicitly configured) PIECEWISE.
            return h[:num_padded_input_tokens]
        if self.data_parallel_size > 1:
            # V1 MTP drafter contract (PR #16669): with DP > 1 the draft batch is DP-uniform (max-token-shaped),
            # so the target must return max-across-DP rows even on ranks holding fewer actual tokens.
            return h[:num_tokens]
        return h[:num_actual_tokens]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | torch.nn.Module:
        # TODO: consider implementing a custom logits computation path for xlite to avoid falling back to the native
        return self.npu_runnable.compute_logits(hidden_states)  # type: ignore[attr-defined]
