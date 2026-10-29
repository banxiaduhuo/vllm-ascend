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
from collections.abc import Sequence
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
from xlite._C import (
    AttnDSA,
    AttnHybridGdn,
    AttnHybridKda,
    AttnMeta,
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
    AttnMetadataRouter,
    WeightGetterConfig,
    XModel,
    XModelConfig,
    _TensorRepr,
    get_dotted_attr,
    get_layer_weights,
    set_dummy_tensor,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v1.xlite.xlite_model_runner import XliteModelRunner as XliteModelRunnerV1
    from vllm_ascend.xlite.xlite_model_runner import XliteModelRunnerV2

    XliteModelRunner: TypeAlias = XliteModelRunnerV1 | XliteModelRunnerV2

XliteForwardResult: TypeAlias = torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]


def _to_int32(t: torch.Tensor) -> torch.Tensor:
    """Cast a device tensor to contiguous int32 (the C++ aliases device tensors zero-copy as int32)."""
    return t.to(torch.int32).contiguous()


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

    from vllm_ascend.attention.attention_v1 import AscendMetadata

    _attn_metadata_type: type | tuple[type, ...] = AscendMetadata
    """The expected type of attention metadata in the forward context for this architecture. Used for runtime checks
    before forwarding. See :meth:`XliteWrapper.__call__` for usage."""
    _attn_meta_version: int = 1
    """The AttnMeta contract version this adapter forwards with: ``1`` = legacy :class:`AttnMeta` host-list path,
    ``2`` = :class:`AttnMetaV2` device-tensor path (``forward_v2``), required when a single legacy metadata cannot
    carry all cache families (e.g. GLM-5.3 kpool). See :meth:`XliteWrapper.__call__`."""
    _supported_architectures: Sequence[str] | str
    """The list of model architecture names (from HuggingFace `config.json` "architectures" field) supported by this
    adapter. Used for automatic adapter selection and registration."""
    _decoder_layer_mlp_module: str = "mlp"
    """The module name used to identify MLP modules (including MoE) in the decoder layers (nn.Module) of the runnable
    model: e.g., in `Glm4MoeForCausalLM`, the MLP modules need to be accessed via `.model.layers[i].mlp`, thus `mlp`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the architecture strategy map and metadata type set."""
        ts = getattr(cls, "_attn_metadata_type", None)
        if ts is None or (not isinstance(ts, type) and not all(isinstance(t, type) for t in ts)):
            raise ValueError(
                f"XliteModel subclass {cls.__name__} must define _attn_metadata_type as a type or a tuple of types."
            )

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
        format = torch_npu.get_npu_format(t)
        return format == torch_npu.Format.FRACTAL_NZ

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
        update_block_size_for_backend() — executor init runs it, but its
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
        max_m = vllm_config.scheduler_config.max_num_batched_tokens
        if not get_ascend_config().xlite_graph_config.full_mode:
            # decode-sized token budget: max_num_seqs * (1 + num_speculative_tokens) instead of max_num_batched_tokens
            n_spec_tokens = getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
            max_m = min(max_m, vllm_config.scheduler_config.max_num_seqs * (n_spec_tokens + 1))
        xlite_config.max_m = (max_m + tp_size - 1) // tp_size * tp_size  # round up to nearest multiple of tp_size
        xlite_config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        xlite_config.max_seq_len = vllm_config.model_config.max_model_len
        xlite_config.block_size = vllm_config.cache_config.block_size

        rope_parameters = getattr(hf_config, "rope_parameters", {})
        xlite_config.deepstack_num_level = len(getattr(self.hf_vision_config, "deepstack_visual_indexes", []))
        xlite_config.mrope_section = rope_parameters.get("mrope_section", [])
        xlite_config.mrope_interleaved = rope_parameters.get("mrope_interleaved", False)
        self.quantization = vllm_config.quant_config is not None

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
    ) -> None:
        """
        Initialize MatMul-related weights with quantization support.

        Args:
            layers (Sequence[torch.nn.Module]): The transformer layers to extract weights from.
            xlite_prefix (str): The prefix for the xlite model attributes to set.
            model_prefix (str): The prefix for the model attributes to look up in each layer.
            mask (Sequence[bool] | None): Optional mask to select specific layers for weight extraction. See
                :meth:`get_layer_weights` for details.
        """
        xlite_model = self.xlite_model
        setattr(xlite_model, xlite_prefix, get_layer_weights(layers, f"{model_prefix}.weight", mask=mask))
        if not self.quantization:
            return

        def set_xlite_attr(xlite_attr: str, layer_attr: str) -> None:
            setattr(xlite_model, xlite_attr, get_layer_weights(layers, layer_attr, mask=mask))

        wt_kwargs: WeightGetterConfig = {"mask": mask, "post_processor": self._transform_deq_scale}
        # A masked probe is polluted by dummy placeholders for missing attrs (they are
        # rebindable non-empty tensors), which would misclassify dynamic-quant
        # checkpoints as static. Probe unmasked first: missing attrs filter out to an
        # empty list, so a non-empty result is a reliable static-quant signal.
        if len(get_layer_weights(layers, f"{model_prefix}.deq_scale")) > 0:  # static quant
            deq_scale = get_layer_weights(layers, f"{model_prefix}.deq_scale", **wt_kwargs)
            setattr(xlite_model, f"{xlite_prefix}_deq_scale", deq_scale)
            set_xlite_attr(f"{xlite_prefix}_input_scale", f"{model_prefix}.aclnn_input_scale_reciprocal")
            set_xlite_attr(f"{xlite_prefix}_input_offset", f"{model_prefix}.aclnn_input_offset")
            set_xlite_attr(f"{xlite_prefix}_quant_bias", f"{model_prefix}.quant_bias")
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


class Qwen35XliteModel(StandardXliteModel):
    """xlite adapter for Qwen3.5: hybrid GatedDeltaNet (KDA) linear attention + GQA full attention.

    Binds the vLLM-sharded weights to the xlite layouts: Gemma-style RMSNorm ``x * (1 + w)`` is
    converted to the standard ``x * w`` (:meth:`_gemma_norm`); the full-attention ``qkv_proj``
    rows ``[q|gate interleaved, k, v]`` are repacked to ``[Q|K|V|Gate]`` (``attn_output_gate``);
    the fused GDN ``in_proj_qkvz``/``in_proj_ba`` rows ``[q|k|v|z]``/``[b|a]`` are split into the
    C++ ``linear_in_proj_qkv/z/b/a`` slots.

    Metadata is assembled into :class:`AttnMetaV2` with ``block_tables = [main paged KV, mamba
    state indices]`` and a single main ``slot_mapping``.
    """

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _attn_metadata_type = (XliteModelBase._attn_metadata_type, GDNAttentionMetadata)
    _attn_meta_version = 2
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

        # Hybrid dispatch (KDA linear attention + GQA full attention).
        xlite_config.attn_type = AttnHybridGdn
        xlite_config.full_attention_interval = getattr(hf_config, "full_attention_interval", 4)
        xlite_config.linear_num_k_heads = hf_config.linear_num_key_heads
        xlite_config.linear_num_v_heads = hf_config.linear_num_value_heads
        xlite_config.linear_key_head_dim = hf_config.linear_key_head_dim
        xlite_config.linear_value_head_dim = hf_config.linear_value_head_dim
        xlite_config.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        xlite_config.qk_norm = True
        xlite_config.attn_output_gate = True
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

    def _get_rope(self, layer_idx: int = 0) -> AscendRotaryEmbedding | None:
        """Probe the first full-attention layer for the rotary embedding (index 0 may be a KDA layer)."""
        layers, _ = self._get_layers_and_model_prefix()
        for i in range(len(layers)):
            if (rope := get_dotted_attr(layers, f"{i}.self_attn.rotary_emb")) is not None:
                return rope
        logger.warning_once("xlite adapter: Rotary embedding not found in the model layers. RoPE will be disabled.")
        return None

    def _build_model(self) -> None:
        # NOTE: does NOT call super()._build_model() — the per-layer slots split by attention type
        # (packed mha_qkv vs the GDN linear-in-proj set) need the empty-placeholder layout the C++
        # init skips placeholder entries on.
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()
        hf_config = self.hf_text_config
        n_layers = xlite_config.n_layers
        tp_size = xlite_config.def_tp_size

        # embed / norm / head (all Qwen3.5 RMSNorms are Gemma-style).
        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = self._gemma_norm(
            get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        )
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)
        xlite_model.attn_norm = [self._gemma_norm(w) for w in get_layer_weights(layers, "input_layernorm.weight")]
        xlite_model.mlp_norm = [
            self._gemma_norm(w) for w in get_layer_weights(layers, "post_attention_layernorm.weight")
        ]

        # Per-layer KDA vs full-attention dispatch from the runnable layer modules (authoritative;
        # layer_type is set on every Qwen3_5DecoderLayer).
        self.kda_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "layer_type", "") == "linear_attention"
        ]
        kda_set = set(self.kda_layers)
        self.full_attn_layers = [i for i in range(n_layers) if i not in kda_set]

        # Full-attention local geometry (qkv_proj rows: [q|gate interleaved, k, v]).
        n_heads_local = max(1, hf_config.num_attention_heads // tp_size)
        n_kv_local = max(1, hf_config.num_key_value_heads // tp_size)
        head_dim = xlite_config.head_dim
        q_rows = n_heads_local * head_dim
        kv_size = n_kv_local * head_dim
        # KDA linear-attention local geometry (qkvz rows: [q|k|v|z] segments, ba rows: [b|a]).
        k_dim = hf_config.linear_key_head_dim
        v_dim = hf_config.linear_value_head_dim
        n_k_local = max(1, hf_config.linear_num_key_heads // tp_size)
        n_v_local = max(1, hf_config.linear_num_value_heads // tp_size)
        kdl, vdl = n_k_local * k_dim, n_v_local * v_dim

        attn_out: list[torch.Tensor] = []
        mha_qkv: list[torch.Tensor] = []
        mha_qkv_bias: list[torch.Tensor] = []
        mha_q_norm: list[torch.Tensor] = []
        mha_k_norm: list[torch.Tensor] = []
        lip_qkv: list[torch.Tensor] = []
        lip_z: list[torch.Tensor] = []
        lip_b: list[torch.Tensor] = []
        lip_a: list[torch.Tensor] = []
        conv1d: list[torch.Tensor] = []
        a_log: list[torch.Tensor] = []
        dt_bias: list[torch.Tensor] = []
        lin_norm: list[torch.Tensor] = []
        out_proj: list[torch.Tensor] = []

        for i, layer in enumerate(layers):
            if i in kda_set:
                la = layer.linear_attn
                qkvz = la.in_proj_qkvz.weight
                ba = la.in_proj_ba.weight
                attn_out.append(_DUMMY_TENSOR)
                mha_qkv.append(_DUMMY_TENSOR)
                if xlite_config.qk_norm:
                    mha_q_norm.append(_DUMMY_TENSOR)
                    mha_k_norm.append(_DUMMY_TENSOR)
                if xlite_config.qkv_bias:
                    mha_qkv_bias.append(_DUMMY_TENSOR)
                lip_qkv.append(qkvz[: 2 * kdl + vdl])
                lip_z.append(qkvz[2 * kdl + vdl :])
                lip_b.append(ba[:n_v_local])
                lip_a.append(ba[n_v_local:])
                conv1d.append(la.conv1d.weight)
                a_log.append(la.A_log)
                dt_bias.append(la.dt_bias)
                lin_norm.append(la.norm.weight)  # KDA gated RMSNorm is standard x*w already
                out_proj.append(la.out_proj.weight)
            else:
                attn = layer.self_attn
                qkv = attn.qkv_proj.weight
                q_gate, k_w, v_w = qkv.split([2 * q_rows, kv_size, kv_size], dim=0)
                qg = q_gate.view(n_heads_local, 2, head_dim, -1)
                # C++ mha_qkv layout: [Q|K|V|Gate].
                mha_qkv.append(torch.cat([qg[:, 0].reshape(q_rows, -1), k_w, v_w, qg[:, 1].reshape(q_rows, -1)], dim=0))
                if attn.qkv_proj.bias is not None:
                    q_gate_b, k_b, v_b = attn.qkv_proj.bias.split([2 * q_rows, kv_size, kv_size], dim=0)
                    bg = q_gate_b.view(n_heads_local, 2, head_dim)
                    mha_qkv_bias.append(torch.cat([bg[:, 0].reshape(-1), k_b, v_b, bg[:, 1].reshape(-1)], dim=0))
                attn_out.append(attn.o_proj.weight)
                mha_q_norm.append(self._gemma_norm(attn.q_norm.weight))
                mha_k_norm.append(self._gemma_norm(attn.k_norm.weight))
                lip_qkv.append(_DUMMY_TENSOR)
                lip_z.append(_DUMMY_TENSOR)
                lip_b.append(_DUMMY_TENSOR)
                lip_a.append(_DUMMY_TENSOR)
                conv1d.append(_DUMMY_TENSOR)
                a_log.append(_DUMMY_TENSOR)
                dt_bias.append(_DUMMY_TENSOR)
                lin_norm.append(_DUMMY_TENSOR)
                out_proj.append(_DUMMY_TENSOR)

        xlite_model.attn_out = attn_out
        xlite_model.mha_qkv = mha_qkv
        if mha_qkv_bias:
            xlite_model.mha_qkv_bias = mha_qkv_bias
            xlite_config.qkv_bias = True
        if xlite_config.qk_norm:
            xlite_model.mha_q_norm = mha_q_norm
            xlite_model.mha_k_norm = mha_k_norm
        xlite_model.linear_in_proj_qkv = lip_qkv
        xlite_model.linear_in_proj_z = lip_z
        xlite_model.linear_in_proj_b = lip_b
        xlite_model.linear_in_proj_a = lip_a
        xlite_model.linear_conv1d = conv1d
        xlite_model.linear_a_log = a_log
        xlite_model.linear_dt_bias = dt_bias
        xlite_model.linear_norm = lin_norm
        xlite_model.linear_out_proj = out_proj

        # Dense MLP slots (MoE layers carry no mlp.gate_up_proj and are skipped by the getter).
        self.init_matmul_weights(layers, "mlp_up_gate", "mlp.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", "mlp.down_proj")

        # MoE slots (MoE layers only; FusedMoE stacks its local experts on dim 0).
        moe_layers = [layer for layer in layers if getattr(layer.mlp, "experts", None) is not None]
        if moe_layers:
            xlite_model.gate = [layer.mlp.gate.weight for layer in moe_layers]
            xlite_model.re_up_gate = [
                w for layer in moe_layers for w in layer.mlp.experts.routed_experts.w13_weight.unbind(0)
            ]
            xlite_model.re_down = [
                w for layer in moe_layers for w in layer.mlp.experts.routed_experts.w2_weight.unbind(0)
            ]
            xlite_config.gate_captured = self.is_tensor_nz(xlite_model.gate[0])
            xlite_config.experts_weight_nz = self.is_tensor_nz(xlite_model.re_up_gate[0])
            shared_experts = [
                layer.mlp.shared_expert for layer in moe_layers if getattr(layer.mlp, "shared_expert", None) is not None
            ]
            if shared_experts:
                xlite_model.se_up_gate = [s.gate_up_proj.weight for s in shared_experts]
                xlite_model.se_down = [s.down_proj.weight for s in shared_experts]
                xlite_model.se_gate = [layer.mlp.shared_expert_gate.weight for layer in moe_layers]
                xlite_config.n_shared_experts = 1

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Split the flat cache list into per-layer family tuples.

        Families are matched by (dtype, ndim) signatures — the fp32 KDA ssm state is the only
        fp32 family; full-attention layers register one fused ``[2, ...]`` tensor (V1) or a pair
        of adjacent 4D k/v views (V2/MRV2 regrouping). Pairing follows the deterministic
        per-layer registration order; optional trailing MTP layers are dropped by the
        per-family counts. KDA layers -> ``(conv, ssm)``; full-attention layers -> ``(k, v)``.
        """
        n_layers = self.xlite_config.n_layers
        kda_set = set(self.kda_layers)
        n_kda, n_full = len(kda_set), n_layers - len(kda_set)

        fulls: list[torch.Tensor] = []
        convs: list[torch.Tensor] = []
        ssms: list[torch.Tensor] = []
        for t in kv_caches:
            if t.dtype == self.dtype:
                if t.dim() == 5 and t.shape[0] == 2:
                    fulls.append(t)  # V1 flat path: fused k/v
                elif t.dim() == 4:
                    fulls.append(t)  # V2/MRV2: per-layer k or v view of the fused parent
                else:
                    convs.append(t)  # KDA conv state (paged or slot-major)
            else:
                ssms.append(t)  # KDA recurrent state (fp32)
        for fam, family_list, expected in (("full", fulls, n_full), ("conv", convs, n_kda), ("ssm", ssms, n_kda)):
            if len(family_list) < expected:
                raise ValueError(f"xlite qwen3.5: {fam} cache count mismatch ({len(family_list)} vs. {expected}).")

        if fulls[0].dim() == 5:
            full_views = [(t[0], t[1]) for t in fulls[:n_full]]
        else:
            if len(fulls) < 2 * n_full:
                raise ValueError(f"xlite qwen3.5: full cache count mismatch ({len(fulls)} vs. {2 * n_full}).")
            full_views = [(fulls[2 * j], fulls[2 * j + 1]) for j in range(n_full)]
        full_caches = {layer: view for layer, view in zip(self.full_attn_layers, full_views)}
        kda_caches = {layer: (conv, ssm) for layer, conv, ssm in zip(self.kda_layers, convs[:n_kda], ssms[:n_kda])}
        result = [kda_caches[i] if i in kda_set else full_caches[i] for i in range(n_layers)]
        logger.info_once("xlite qwen3.5: registered kv caches (flat): %s", _TensorRepr(result, max_it=None))
        return result

    def build_attn_meta_v2(self, attn_metadata: dict[str, Any], positions: torch.Tensor) -> tuple[Any, int]:
        """Assemble the Qwen3.5 :class:`AttnMetaV2` with family-slot metadata.

        Full attention (``AscendMetadata``) supplies lens/cached_lens/query_start_loc and the
        shared paged KV block table. KDA (``GDNAttentionMetadata``) supplies
        ``non_spec_state_indices_tensor`` (the mamba state indices, i.e. ``block_table[:, 0]``),
        deduplicated into one ``[num_reqs, 1]`` entry per mamba cache group — block ids ARE the
        state indices, mirroring the GLM5 family slot layout.

        Returns:
            tuple[Any, int]: ``(AttnMetaV2, num_actual_tokens)``; ``(None, 0)`` for an empty batch.
        """
        full_meta, _ = AttnMetadataRouter.pick(attn_metadata, self.AscendMetadata, self.GDNAttentionMetadata)
        num_actual_tokens = int(full_meta.num_actual_tokens)
        if num_actual_tokens == 0:
            return None, 0  # empty batch: the wrapper skips the C++ forward

        i32 = _to_int32

        # AscendMetadata.query_start_loc is the 0-prefixed cumulative query boundary tensor.
        query_start_loc = i32(full_meta.query_start_loc)
        lens = i32(query_start_loc[1:] - query_start_loc[:-1])
        num_reqs = lens.numel()
        # MRV2 hosts seq_lens on CPU; prefer the persistent device-side mirror (seq_lens_gpu,
        # int32, refreshed in place every step) so the clamp stays on NPU.
        seq_lens = full_meta.seq_lens_gpu
        if seq_lens is None:
            seq_lens = full_meta.seq_lens.to(query_start_loc.device)
        cached_lens = torch.clamp(i32(seq_lens[:num_reqs]) - lens, min=0)

        # vLLM splits the KDA layers round-robin into mamba cache groups whose tensors alias
        # the same regions from byte 0, so each group needs its own state indices: dedup the
        # per-layer GDNAttentionMetadata into tables for blockTables[1..n] (C++ maps KDA
        # position p to table 1 + p % n).
        meta_by_layer = {
            int(m.group(1)): v
            for name, v in attn_metadata.items()
            if isinstance(v, self.GDNAttentionMetadata) and (m := re.search(r"layers\.(\d+)\.", name))
        }
        tables: list[Any] = []
        tid_by_meta: dict[int, int] = {}
        of_layer: list[int] = []
        for layer in self.kda_layers:
            v = meta_by_layer[layer]
            tid = tid_by_meta.get(id(v))
            if tid is None:
                tid = tid_by_meta[id(v)] = len(tables)
                tables.append(v)
            of_layer.append(tid)
        kda_tables = [i32(v.non_spec_state_indices_tensor).reshape(-1, 1) for v in tables]
        if any(t.shape[0] < num_reqs for t in kda_tables):
            raise ValueError(f"xlite qwen3.5: request-count mismatch across metadata families (main={num_reqs}).")
        if len(tables) > 1 and of_layer != [p % len(tables) for p in range(len(of_layer))]:
            raise ValueError(
                f"xlite qwen3.5: mamba group assignment {of_layer} is not the round-robin "
                f"{[p % len(tables) for p in range(len(of_layer))]} the C++ mapping assumes."
            )

        pos: torch.Tensor = torch.atleast_2d(positions)[:, :num_actual_tokens].contiguous()

        meta = AttnMetaV2()
        meta.lens_cpu = lens.tolist()
        meta.cached_lens_cpu = cached_lens.tolist()
        meta.lens = lens
        meta.cached_lens = cached_lens
        meta.query_start_loc = query_start_loc
        meta.positions = pos.reshape(-1).reshape(pos.shape)  # int64, rematerialized with correct stride
        # Family slots: [0] main paged KV, [1..] KDA mamba state indices — one table per
        # mamba cache group (see the per-group mapping above).
        meta.block_tables = [i32(full_meta.block_tables[:num_reqs]), *kda_tables]
        meta.slot_mapping = [i32(full_meta.slot_mapping)]
        return meta, num_actual_tokens


class DeepseekV3XliteModel(Glm4MoeXliteModel):
    """xlite adapter for DeepseekV3 MoE architectures with MLA attention."""

    from vllm_ascend.attention.mla_v1 import AscendMLAMetadata

    _attn_metadata_type = AscendMLAMetadata  # type: ignore[assignment]
    _supported_architectures = ["DeepseekV3ForCausalLM"]

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

    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    _attn_metadata_type = AscendSFAMetadata  # type: ignore[assignment]
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


class Glm5NextXliteModel(DeepseekV32XliteModel):
    """xlite adapter for GLM-5.3-Flash (glm5_next): hybrid KDA linear attention + NoPE sparse MLA + MHC + MoE.

    Reuses the DeepseekV32 DSA plumbing (AscendSFAMetadata, indexer cache slots) but rebuilds the
    config/weight binding for the two attention families (KDA ``linear_attention`` with fused
    ``in_proj_qkvbfg_a``, sparse MLA with per-layer kpool indexer) and the MHC residual streams.

    Metadata is assembled into :class:`AttnMetaV2` from type-picked families; KV caches keep the
    universal flat path (:meth:`extract_kv_cache`), with the family router keyed on
    (dtype, ndim, last-dim) since the GLM5 families collide on the last dim alone.
    """

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_ascend.attention.indexer_kpool import (  # noqa: F401  (re-exported for tests)
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )

    # The kpool path consumes the main SFA metadata plus the tail-ring cache metadata and the GDN
    # (KDA) mamba slot indices; the indexer-pool metadata is derivable in C++ (shared block ids +
    # compress_ratios) and is not consumed here. Declaring the full trio + GDN keeps the runner's
    # backend-compatibility check quiet.
    _attn_metadata_type = (
        DeepseekV32XliteModel._attn_metadata_type,
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
        GDNAttentionMetadata,
    )
    _attn_meta_version = 2

    _supported_architectures = ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]

    def _build_model_config(self) -> None:
        # Skip the DeepseekV3/V32 chain: GLM5 has no standard qkv_proj, its indexer fields fold
        # differently, and KDA/MLA dispatch is explicit. The Standard base plus the GLM5-specific
        # fields below cover the full config surface.
        # Glm5NextConfig declares head_dim=None (MLA latent layout, no per-head dim), but the
        # Standard base does `head_dim**-0.5` unconditionally. Substitute a placeholder for the
        # base's arithmetic (overwritten below) and restore the original afterwards.
        saved_head_dim = getattr(self.hf_text_config, "head_dim", None)
        if not saved_head_dim:  # None or 0 (MLA latent layout has no per-head dim)
            self.hf_text_config.head_dim = 1
        StandardXliteModel._build_model_config(self)
        if saved_head_dim is not None and not saved_head_dim:
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

        # MHC residual mixing (Glm5NextConfig folds hc_mult/hc_sinkhorn_iters into the aliases).
        xlite_config.hc_mult = hf_config.mhc_num_residual_streams
        xlite_config.hc_sinkhorn_iters = hf_config.mhc_sinkhorn_iterations or 20
        xlite_config.hc_eps = hf_config.hc_eps

        # KDA linear-attention dims (q/k/v share the same head layout).
        xlite_config.linear_num_k_heads = hf_config.linear_num_heads
        xlite_config.linear_num_v_heads = hf_config.linear_num_heads
        xlite_config.linear_key_head_dim = hf_config.linear_head_dim
        xlite_config.linear_value_head_dim = hf_config.linear_head_dim
        xlite_config.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
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

        # Per-layer KDA-vs-MLA dispatch from layer_types. Pad a short layer_types (e.g. when the
        # MTP layer is not listed) with sparse-MLA, matching the tail-layer layout. Python-side
        # only: drives the weight masks below and the KV cache family assembly.
        layer_types: list[str] = list(getattr(hf_config, "layer_types", []) or [])
        if len(layer_types) < xlite_config.n_layers:
            layer_types += ["deepseek_sparse_attention"] * (xlite_config.n_layers - len(layer_types))
        self.kda_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        self.full_attn_layers = [i for i, t in enumerate(layer_types) if t != "linear_attention"]

        # MoE (W8A8 routed/shared experts, BF16 gate; dense layers use standard MLP slots).
        xlite_config.n_dense_layers = getattr(hf_config, "first_k_dense_replace", 0)
        xlite_config.n_routed_experts = hf_config.n_routed_experts
        xlite_config.n_shared_experts = hf_config.n_shared_experts
        xlite_config.n_act_experts = hf_config.num_experts_per_token
        xlite_config.n_expert_groups = getattr(hf_config, "n_group", 1)
        xlite_config.n_limited_groups = getattr(hf_config, "topk_group", 1)
        xlite_config.moe_intermediate_size = hf_config.moe_intermediate_size
        xlite_config.intermediate_size = hf_config.intermediate_size
        xlite_config.norm_topk_prob = True
        xlite_config.scoring_func = ScoringFuncSigmoid
        xlite_config.route_scale = getattr(hf_config, "routed_scaling_factor", 1.0)
        xlite_config.swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0) or 0.0
        xlite_config.gate_captured = False  # GLM5 router gate is BF16 (never NZ-captured)

    def _build_model(self) -> None:
        # NOTE: does NOT call super()._build_model() — GLM5 has no self_attn.qkv_proj (standard MHA
        # path) nor the Deepseek MLA/indexer layout; all GLM5-specific slots (KDA fused split, MLA,
        # indexer, MHC, MoE) are bound here against empty(1) placeholders on non-applicable layers
        # (the C++ init skips placeholder entries).
        xlite_model, xlite_config = self.xlite_model, self.xlite_config
        layers, model_prefix = self._get_layers_and_model_prefix()
        hf_config = self.hf_text_config
        n_layers = xlite_config.n_layers
        tp_size = xlite_config.def_tp_size

        # embed / norm / head / layer norms
        xlite_model.embed = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.embed_tokens.weight", raises=True)
        xlite_model.norm = get_dotted_attr(self.npu_runnable, f"{model_prefix}model.norm.weight", raises=True)
        if getattr(hf_config, "tie_word_embeddings", False):
            xlite_model.head = xlite_model.embed
        else:
            xlite_model.head = get_dotted_attr(self.npu_runnable, f"{model_prefix}lm_head.weight", raises=True)
        xlite_model.attn_norm = get_layer_weights(layers, "input_layernorm.weight")
        xlite_model.mlp_norm = get_layer_weights(layers, "post_attention_layernorm.weight")

        # Per-layer KDA vs MLA masks.
        kda_set = set(self.kda_layers)
        kda_mask = [i in kda_set for i in range(n_layers)]
        mla_mask = [not m for m in kda_mask]

        # --- KDA weights (BF16; AscendC KDA runs unquantized) ---
        # Fused in_proj_qkvbfg_a row shards: [q, k, v (proj each), b (heads), f_a, g_a (head_dim,
        # replicated)]. The 2D row slice must no-op on the 1D _DUMMY_TENSOR substituted for
        # masked-out layers.
        proj = hf_config.linear_head_dim * hf_config.linear_num_heads
        lp = proj // tp_size  # local projection (q/k/v per rank)
        lh = hf_config.linear_num_heads // tp_size  # local heads (b per rank)
        hd = hf_config.linear_head_dim
        _slices = {
            "glm5_q_proj": (0, lp),
            "glm5_k_proj": (lp, lp),
            "glm5_v_proj": (2 * lp, lp),
            "glm5_b_proj": (3 * lp, lh),
            "glm5_f_a_proj": (3 * lp + lh, hd),
            "glm5_g_a_proj": (3 * lp + lh + hd, hd),
        }
        for slot, (start, length) in _slices.items():
            setattr(
                xlite_model,
                slot,
                get_layer_weights(
                    layers,
                    "self_attn.in_proj_qkvbfg_a.weight",
                    mask=kda_mask,
                    post_processor=(
                        lambda s, ln: (lambda w: w[s : s + ln, :].contiguous() if w.ndim >= 2 else w)  # noqa: E731
                    )(start, length),
                ),
            )
        self.init_matmul_weights(layers, "glm5_f_b_proj", "self_attn.f_b_proj", mask=kda_mask)
        self.init_matmul_weights(layers, "glm5_g_b_proj", "self_attn.g_b_proj", mask=kda_mask)
        self.init_matmul_weights(layers, "glm5_wo", "self_attn.o_proj", mask=kda_mask)
        xlite_model.glm5_q_conv1d = get_layer_weights(layers, "self_attn.q_conv1d.weight", mask=kda_mask)
        xlite_model.glm5_k_conv1d = get_layer_weights(layers, "self_attn.k_conv1d.weight", mask=kda_mask)
        xlite_model.glm5_v_conv1d = get_layer_weights(layers, "self_attn.v_conv1d.weight", mask=kda_mask)
        # Conv1d weights are stored fp32 (unquantized FLOAT rows); the C++ kernel
        # requires weight/activation dtype match — native casts to the activation
        # dtype once after load (kda.py _merged_conv_weight), mirror that here.
        # NOTE: pybind def_readwrite returns a COPY of the C++ vector, so the cast
        # list must be written back via setattr (in-place item assignment is lost).
        for conv_name in ("glm5_q_conv1d", "glm5_k_conv1d", "glm5_v_conv1d"):
            conv_w = getattr(xlite_model, conv_name)
            if conv_w and conv_w[0].dtype != self.dtype:
                setattr(xlite_model, conv_name, [w.to(self.dtype) for w in conv_w])
        # rc2 stores A_log as [1, 1, local_heads, 1]; the C++ reference consumes the 1-D
        # [local_heads] form (tests/models/glm5_next.py: a.A_log[_sh]).
        xlite_model.glm5_a_log = get_layer_weights(
            layers, "self_attn.A_log", mask=kda_mask, post_processor=lambda w: w.reshape(-1) if w.ndim > 1 else w
        )
        xlite_model.glm5_dt_bias = get_layer_weights(layers, "self_attn.dt_bias", mask=kda_mask)
        xlite_model.glm5_o_norm = get_layer_weights(layers, "self_attn.o_norm.weight", mask=kda_mask)

        # --- sparse MLA weights (full-attn layers) ---
        self.init_matmul_weights(layers, "mla_qkv_a", "self_attn.fused_qkv_a_proj", mask=mla_mask)
        self.init_matmul_weights(layers, "mla_q_b", "self_attn.q_b_proj", mask=mla_mask)
        xlite_model.mla_q_norm = get_layer_weights(layers, "self_attn.q_a_layernorm.weight", mask=mla_mask)
        xlite_model.mla_kv_norm = get_layer_weights(layers, "self_attn.kv_a_layernorm.weight", mask=mla_mask)
        xlite_model.mla_wuv = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UV", mask=mla_mask)
        xlite_model.mla_wuk_t = get_layer_weights(layers, "self_attn.mla_attn.mla_attn.impl.W_UK_T", mask=mla_mask)
        xlite_config.mla_wuv_weight_nz = bool(wuv_lst := xlite_model.mla_wuv) and self.is_tensor_nz(
            next((w for w in wuv_lst if w.ndim >= 2), wuv_lst[0])
        )
        xlite_config.mla_wukt_weight_nz = bool(wuk_lst := xlite_model.mla_wuk_t) and self.is_tensor_nz(
            next((w for w in wuk_lst if w.ndim >= 2), wuk_lst[0])
        )
        self.init_matmul_weights(layers, "attn_out", "self_attn.o_proj", mask=mla_mask)

        # --- kpool indexer (MLA layers) ---
        # native Glm5NextKPoolIndexerBackend.process_weights_after_loading casts
        # wk_weights_proj / kpool_compress_gate / k_norm (weight+bias) to fp32 and
        # runs F.linear/LayerNorm on fp32; the C++ matmul/LayerNorm consume their
        # fp32-weight branches, so the bindings must be fp32 too. The empty
        # _DUMMY_TENSOR placeholders are skipped by the C++ init and float() on
        # an empty tensor is a no-op, so the cast applies unconditionally.
        idx_hd = hf_config.index_head_dim
        xlite_model.glm5_idx_wq_b = get_layer_weights(layers, "self_attn.indexer.wq_b.weight", mask=mla_mask)
        # wk_weights_proj is fused [wk (idx_hd rows); weights_proj] — split per the reference.
        xlite_model.glm5_idx_wk = get_layer_weights(
            layers,
            "self_attn.indexer.wk_weights_proj.weight",
            mask=mla_mask,
            post_processor=lambda w: (w[:idx_hd, :].contiguous() if w.ndim >= 2 else w).to(torch.float32),
        )
        xlite_model.glm5_idx_weights_proj = get_layer_weights(
            layers,
            "self_attn.indexer.wk_weights_proj.weight",
            mask=mla_mask,
            post_processor=lambda w: (w[idx_hd:, :].t().contiguous() if w.ndim >= 2 else w).to(torch.float32),
        )
        xlite_model.glm5_idx_k_norm = get_layer_weights(
            layers, "self_attn.indexer.k_norm.weight", mask=mla_mask, post_processor=lambda t: t.to(torch.float32)
        )
        xlite_model.glm5_idx_k_norm_bias = get_layer_weights(
            layers, "self_attn.indexer.k_norm.bias", mask=mla_mask, post_processor=lambda t: t.to(torch.float32)
        )
        xlite_model.glm5_idx_kpool_gate = get_layer_weights(
            layers,
            "self_attn.indexer.index_kpool_compress_gate",
            mask=mla_mask,
            post_processor=lambda t: t.to(torch.float32),
        )
        # native validates the APE as fp32 [kpool, head_dim]; cast defensively.
        xlite_model.glm5_idx_kpool_ape = get_layer_weights(
            layers,
            "self_attn.indexer.index_kpool_compress_ape",
            mask=mla_mask,
            post_processor=lambda t: t.to(torch.float32),
        )

        # --- MHC residual mixing params (fp32; every layer) ---
        for slot in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            setattr(xlite_model, slot, get_layer_weights(layers, slot, post_processor=lambda t: t.to(torch.float32)))

        # --- dense MLP + MoE (reuse Standard patterns) ---
        mlp_prefix = self._decoder_layer_mlp_module
        self.init_matmul_weights(layers, "mlp_up_gate", f"{mlp_prefix}.gate_up_proj")
        self.init_matmul_weights(layers, "mlp_down", f"{mlp_prefix}.down_proj")
        xlite_model.gate = get_layer_weights(layers, f"{mlp_prefix}.gate.weight")
        xlite_model.gate_bias = get_layer_weights(
            layers, f"{mlp_prefix}.gate.e_score_correction_bias", post_processor=lambda b: b.to(torch.float32)
        )
        self.init_matmul_weights(layers, "se_up_gate", f"{mlp_prefix}.shared_experts.gate_up_proj")
        self.init_matmul_weights(layers, "se_down", f"{mlp_prefix}.shared_experts.down_proj")
        re_prefix = f"{mlp_prefix}.experts.routed_experts"
        re_kwargs: WeightGetterConfig = {"secondary_flattening": f"{re_prefix}.local_num_experts"}
        xlite_model.re_up_gate = get_layer_weights(layers, f"{re_prefix}.w13_weight", **re_kwargs)
        xlite_model.re_down = get_layer_weights(layers, f"{re_prefix}.w2_weight", **re_kwargs)

        if not self.quantization:
            return

        # Attention quant applies to the sparse-MLA latent projections (KDA stays BF16);
        # sample NZ-ness from the first real (non-placeholder) entry.
        self.xlite_config.quant_attn_weight_nz = bool(wt_lst := xlite_model.mla_qkv_a) and self.is_tensor_nz(
            next((w for w in wt_lst if w.ndim >= 2), wt_lst[0])
        )
        self.xlite_config.quant_attn_weight_transpose = True
        xlite_config.experts_weight_nz = bool(xlite_model.re_up_gate) and self.is_tensor_nz(xlite_model.re_up_gate[0])
        re_kwargs["post_processor"] = self._transform_deq_scale
        xlite_model.re_up_gate_scale = get_layer_weights(layers, f"{re_prefix}.w13_weight_scale", **re_kwargs)
        xlite_model.re_down_scale = get_layer_weights(layers, f"{re_prefix}.w2_weight_scale", **re_kwargs)

    def extract_kv_cache(self, kv_caches: list[torch.Tensor], /) -> list[tuple[torch.Tensor, ...]]:
        """Split the flat cache list into per-layer GLM5 family tuples.

        The flat list arrives in per-layer registration order; each sparse-MLA layer contributes
        (main, indexer, tail) tensors and each KDA layer its (conv, ssm) pair. GLM5 families
        collide on the last dim, so each family is matched by a (dtype, ndim, last-dim) signature
        derived from the model config. KDA layers -> ``(conv, ssm)``; sparse-MLA layers ->
        ``(main, indexer, tail)``, both in model layer order.
        """
        xcfg = self.xlite_config
        n_layers = xcfg.n_layers
        kda_set = set(self.kda_layers)
        n_kda, n_mla = len(kda_set), n_layers - len(kda_set)
        if xcfg.kv_lora_rank == xcfg.index_head_dim:
            raise RuntimeError(
                "xlite glm5: kv_lora_rank == index_head_dim; the flat cache list cannot disambiguate "
                "the main/indexer families."
            )

        def family_of(t: torch.Tensor) -> str:
            if t.dtype == self.dtype:
                if t.dim() == 3:  # KDA conv state [slots, 3*(K-1), local_proj]
                    return "conv"
                if t.shape[-1] == xcfg.kv_lora_rank:
                    return "main"
                if t.shape[-1] == xcfg.index_head_dim:
                    return "indexer"
            elif t.dim() == 4 and t.shape[-2] == xcfg.linear_key_head_dim and t.shape[-1] == xcfg.linear_key_head_dim:
                return "ssm"  # KDA recurrent state [slots, heads, head_dim, head_dim]
            else:
                return "tail"  # fp32 kpool tail ring; its folding is confirmed on first run
            return "unknown"

        families: dict[str, list[torch.Tensor]] = {"main": [], "indexer": [], "tail": [], "conv": [], "ssm": []}
        for t in kv_caches:
            fam = family_of(t)
            if fam == "unknown":
                raise ValueError(
                    f"xlite glm5: unclassifiable cache tensor {tuple(t.shape)} {t.dtype} in the flat list."
                )
            families[fam].append(t)
        # Under-count is the real error signal; over-count comes from trailing MTP layers.
        for fam, expected in (("main", n_mla), ("indexer", n_mla), ("tail", n_mla), ("conv", n_kda), ("ssm", n_kda)):
            if (got := len(families[fam])) < expected:
                raise ValueError(f"xlite glm5: {fam} cache count mismatch ({got} vs. {expected}).")
            families[fam] = families[fam][:expected]

        mains, indexers, tails = families["main"], families["indexer"], families["tail"]
        convs, ssms = families["conv"], families["ssm"]
        mla_caches = {
            layer: (
                self._as_block_major(main, xcfg.kv_lora_rank, "main MLA", layer),
                self._as_block_major(indexer, xcfg.index_head_dim, "indexer kpool", layer).unsqueeze(2),
                tail,
            )
            for layer, main, indexer, tail in zip(self.full_attn_layers, mains, indexers, tails)
        }
        kda_caches = {
            layer: self._validate_kda_pool(conv, ssm, layer) for layer, conv, ssm in zip(self.kda_layers, convs, ssms)
        }
        result = [kda_caches[i] if i in kda_set else mla_caches[i] for i in range(n_layers)]
        logger.info_once("xlite glm5: registered kv caches (flat): %s", _TensorRepr(result, max_it=None))
        return result

    @staticmethod
    def _as_block_major(view: Any, width: int, family: str, layer: int) -> torch.Tensor:
        """Squeeze the singleton head dim off the 4D ``[B, H=1, N, C]`` cache view to ``[B, N, C]``.

        Views may be non-contiguous (caches register with ``check_is_contiguous=False``); only
        the family-identifying last-dim ``width`` is validated here.
        """
        if isinstance(view, (list, tuple)) or not isinstance(view, torch.Tensor):
            raise ValueError(f"xlite glm5: layer {layer} {family} cache view is not a single tensor: {view!r}")
        t = view.squeeze(1) if view.dim() == 4 and view.shape[1] == 1 else view
        if t.dim() != 3 or t.shape[-1] != width:
            raise ValueError(
                f"xlite glm5: layer {layer} {family} cache view {tuple(view.shape)} does not collapse to "
                f"[blocks, states, {width}]"
            )
        return t

    def build_attn_meta_v2(self, attn_metadata: dict[str, Any], positions: torch.Tensor) -> tuple[Any, int]:
        """Assemble the GLM5 kpool :class:`AttnMetaV2`.

        Families are picked type-driven (:meth:`AttnMetadataRouter.pick`): main SFA supplies
        lens/cached_lens/query_start_loc and the shared block table (main and indexer pools share
        block ids). GDN (KDA) per-layer metadatas are deduplicated into one ``[num_reqs, 1]``
        table per mamba cache group for ``block_tables[1..n]`` (C++ maps KDA position p to table
        ``1 + p % n``); the tail ring's ``block_table`` is the last slot and never takes part in
        that mapping. ``AscendSFAMetadata.cum_query_lens`` lacks the leading zero, so
        query_start_loc is rebuilt with ``torch.cumsum(..., dtype=torch.int32)``.

        Returns:
            tuple[Any, int]: ``(AttnMetaV2, num_actual_tokens)``; ``(None, 0)`` for an empty batch.
        """
        sfa_meta, tail_meta, _ = AttnMetadataRouter.pick(
            attn_metadata,
            self.AscendSFAMetadata,
            self.AscendIndexerKPoolTailMetadata,
            self.GDNAttentionMetadata,
        )
        num_actual_tokens = int(sfa_meta.num_actual_tokens)
        if num_actual_tokens == 0:
            return None, 0  # empty batch: the wrapper skips the C++ forward

        i32 = _to_int32

        lens = i32(sfa_meta.cum_query_lens)
        num_reqs = lens.numel()
        query_start_loc = torch.cat((lens.new_zeros(1), torch.cumsum(lens, 0, dtype=torch.int32)))
        cached_lens = torch.clamp(i32(sfa_meta.seq_lens[:num_reqs]) - lens, min=0)

        # vLLM splits the KDA layers round-robin into mamba cache groups whose tensors alias
        # the same regions from byte 0, so each group needs its own state indices.
        meta_by_layer = {
            int(m.group(1)): v
            for name, v in attn_metadata.items()
            if isinstance(v, self.GDNAttentionMetadata) and (m := re.search(r"layers\.(\d+)\.", name))
        }
        tables: list[Any] = []
        tid_by_meta: dict[int, int] = {}
        of_layer: list[int] = []
        for layer in self.kda_layers:
            v = meta_by_layer[layer]
            tid = tid_by_meta.get(id(v))
            if tid is None:
                tid = tid_by_meta[id(v)] = len(tables)
                tables.append(v)
            of_layer.append(tid)
        kda_tables = [i32(v.non_spec_state_indices_tensor).reshape(-1, 1) for v in tables]
        tail_tables = i32(tail_meta.block_table)
        if any(t.shape[0] < num_reqs for t in kda_tables) or tail_tables.shape[0] != num_reqs:
            raise ValueError(
                f"xlite glm5: request-count mismatch across metadata families "
                f"(main={num_reqs}, kda={[t.shape[0] for t in kda_tables]}, tail={tail_tables.shape[0]})."
            )
        if len(tables) > 1 and of_layer != [p % len(tables) for p in range(len(of_layer))]:
            raise ValueError(
                f"xlite glm5: mamba group assignment {of_layer} is not the round-robin "
                f"{[p % len(tables) for p in range(len(of_layer))]} the C++ mapping assumes."
            )

        if positions.dim() != 1:
            raise ValueError(f"xlite glm5: positions must be 1D [tokens], got {tuple(positions.shape)}.")
        pos = positions[:num_actual_tokens].contiguous()

        meta = AttnMetaV2()
        meta.lens_cpu = lens.tolist()
        meta.cached_lens_cpu = cached_lens.tolist()
        meta.lens = lens
        meta.cached_lens = cached_lens
        meta.query_start_loc = query_start_loc
        meta.positions = pos.reshape(-1).reshape(pos.shape)  # int64, rematerialized with correct stride
        # Family slots: [0] main KV (shared by the indexer pool), [1..n] KDA mamba state indices
        # (one per mamba cache group), [n+1] tail ring.
        meta.block_tables = [i32(sfa_meta.block_table[:num_reqs]), *kda_tables, tail_tables]
        meta.slot_mapping = [i32(sfa_meta.slot_mapping)]
        return meta, num_actual_tokens

    def _validate_kda_pool(
        self, conv: torch.Tensor, ssm: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate the native (paged) mamba pool pair against the C++ 2-slot KDA contract.

        conv ``[slots, 3*(d_conv-1), localProj]`` bf16 (page = contiguous q|k|v history
        windows), ssm ``[slots, local_heads, head_dim, head_dim]`` fp32. The C++ kernels index
        pool rows per request via ``block_tables[1]``; fresh-request rows are zeroed host-side
        (semantic equivalent of ``has_initial_state = context_lens > 0``).
        """
        cfg = self.xlite_config
        local_heads = cfg.linear_num_k_heads // cfg.def_tp_size
        head_dim = cfg.linear_key_head_dim
        conv_rows = 3 * (cfg.linear_conv_kernel_dim - 1)
        proj = local_heads * head_dim
        if conv.dim() != 3 or conv.shape[1] != conv_rows or conv.shape[2] != proj:
            raise ValueError(
                f"xlite glm5: layer {layer} native conv pool {tuple(conv.shape)} does not match "
                f"[slots, 3*(d_conv-1)={conv_rows}, {proj}]"
            )
        if ssm.dim() != 4 or tuple(ssm.shape[1:]) != (local_heads, head_dim, head_dim):
            raise ValueError(
                f"xlite glm5: layer {layer} native ssm pool {tuple(ssm.shape)} does not match "
                f"[slots, {local_heads}, {head_dim}, {head_dim}]"
            )
        if conv.shape[0] != ssm.shape[0] or conv.shape[0] == 0:
            raise ValueError(
                f"xlite glm5: layer {layer} conv/ssm pool slot counts differ or are empty: "
                f"{conv.shape[0]} vs {ssm.shape[0]}"
            )
        if ssm.dtype != torch.float32:
            raise ValueError(f"xlite glm5: layer {layer} ssm pool must be fp32, got {ssm.dtype}")
        return conv, ssm


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

        set_dummy_tensor(torch.empty(1, device=device, dtype=vllm_config.model_config.dtype))
        self.adapter_xlite_model = get_adapter_xlite_model(npu_runner, vllm_config)  # python adapter instance
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

    def register_kv_caches(self, kv_caches: list[torch.Tensor]) -> None:
        """Register KV cache references used by xlite runtime.

        Args:
            kv_caches (list[torch.Tensor]): Runtime KV cache handles or tensors.
        """
        self.kv_caches = self.adapter_xlite_model.extract_kv_cache(kv_caches)

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
        attn_metadata: Any = forward_context.attn_metadata
        attn_metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
        # V2 adapters (GLM5 kpool) build an AttnMetaV2 from the full per-layer metadata dict
        # (type-driven pick below); V1 adapters reduce to a single metadata object first — the
        # "layers.0" key is a KDA layer on GLM5, so the reduction must not happen there.
        if getattr(self.adapter_xlite_model, "_attn_meta_version", 1) != 2:
            attn_metadata = attn_metadata.get("model.layers.0.self_attn.attn", next(iter(attn_metadata.values()), None))
        # Since v0.28.0, batches are routed by token count (max across DP ranks) instead of the batch attention
        # state: batches within the preallocated budget run on xlite, others fall back to the native acl runnable.
        num_tokens: int
        if (num_tokens := _EXTRA_CTX.max_tokens_across_dp) is None:
            if (dp_metadata := forward_context.dp_metadata) is not None:
                num_tokens = dp_metadata.num_tokens_across_dp_cpu.max().item()
            else:
                num_tokens = attn_metadata.num_actual_tokens
        if num_tokens > self.max_tokens:
            if self.full_mode:
                logger.warning_once(
                    "xlite: token number exceeded expected limit (%d >%d), consider opening an issue at %s",
                    num_tokens,
                    self.max_tokens,
                    "https://atomgit.com/openeuler/GVirt",
                )
            return self.npu_runnable(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)

        stream = torch.npu.current_stream().npu_stream
        if getattr(self.adapter_xlite_model, "_attn_meta_version", 1) == 2:
            attn_meta, num_actual_tokens = self.adapter_xlite_model.build_attn_meta_v2(attn_metadata, positions)
            if attn_meta is None:
                # Empty batch (async scheduling emits 0-token steps): skip the C++ forward entirely.
                return self.hidden_states[:0]
            forward, forward_with_inputs_embeds = (
                self.xlite_model.forward_v2,
                self.xlite_model.forward_with_inputs_embeds_v2,
            )
            # forward_v2 takes the RoPE cache as a tensor list (the V1 path takes the bare cache).
            freqs_cis = [self.freq_cis]
        else:
            attn_metadata_router = AttnMetadataRouter(attn_metadata=attn_metadata, device="cpu")
            num_actual_tokens = attn_metadata_router.num_actual_tokens
            seq_lens = attn_metadata_router.seq_lens
            cum_query_lens = attn_metadata_router.cu_query_lens[-seq_lens.size(0) :]
            query_lens = torch.diff(cum_query_lens, prepend=seq_lens.new_zeros(1))
            cached_lens = torch.clamp(seq_lens - query_lens, min=0)
            block_tables = attn_metadata_router.block_tables
            num_reqs = max(1, (block_tables[:, 0] != 0).sum().item())  # number of non-empty requests

            attn_meta = AttnMeta()
            attn_meta.lens = query_lens[:num_reqs].tolist()
            attn_meta.cached_lens = cached_lens[:num_reqs].tolist()
            attn_meta.block_tables_cpu = block_tables[:num_reqs].tolist()
            pos: torch.Tensor = torch.atleast_2d(positions)[:, :num_actual_tokens].contiguous()
            attn_meta.positions = pos.reshape(-1).reshape(pos.shape)  # rematerialize with correct stride
            forward, forward_with_inputs_embeds = (
                self.xlite_model.forward,
                self.xlite_model.forward_with_inputs_embeds,
            )
            freqs_cis = self.freq_cis

        # under DP, `num_tokens` is the max number of tokens across all DP ranks for data alignment
        h = self.hidden_states[:num_tokens]
        input_ids = cast(torch.Tensor, input_ids)
        if inputs_embeds is None or inputs_embeds.size(0) == 0:
            forward(
                rt=self.xlite_rt,
                input=input_ids[:num_actual_tokens],
                attn_meta=attn_meta,
                kv_cache=self.kv_caches,
                freqs_cis=freqs_cis,
                output=h,
                curr_stream=stream,
            )
        else:
            inputs_embeds = inputs_embeds[:num_actual_tokens]
            deepstack_input_embeds = getattr(self.npu_runnable, "deepstack_input_embeds", [])
            xlite_deepstack_input_embeds = [
                deepstack_input[: inputs_embeds.size(0)] for deepstack_input in deepstack_input_embeds
            ]
            forward_with_inputs_embeds(
                rt=self.xlite_rt,
                input=inputs_embeds,
                attn_meta=attn_meta,
                kv_cache=self.kv_caches,
                # forward_with_inputs_embeds_v2 binds freqs_cis as a single tensor (unlike the
                # tensor-list contract of forward_v2); the V1 path takes the bare cache too.
                freqs_cis=freqs_cis[0] if isinstance(freqs_cis, list) else freqs_cis,
                output=h,
                curr_stream=stream,
                deepstack_input=xlite_deepstack_input_embeds,
            )
            if deepstack_input_embeds and callable(
                _clear_stack := getattr(self.npu_runnable, "_clear_deepstack_input_embeds", None)
            ):
                _clear_stack(inputs_embeds.size(0))
        return h[:num_actual_tokens]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | torch.nn.Module:
        # TODO: consider implementing a custom logits computation path for xlite to avoid falling back to the native
        return self.npu_runnable.compute_logits(hidden_states)  # type: ignore[attr-defined]
