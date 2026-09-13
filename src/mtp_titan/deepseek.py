from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common import Linear, RMSNorm
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module, ModuleDict

from .targets import mtp_labels


def deepseek_head_weights(
    num_modules: int, mtp_loss_weight: float
) -> tuple[float, ...]:
    """DeepSeek-V3 §2.2, eq. 25: the main model keeps weight 1, the D depths
    share λ/D."""
    if num_modules == 0:
        return (1.0,)
    return (1.0,) + (mtp_loss_weight / num_modules,) * num_modules


class DeepSeekMtpModule(Module):
    """DeepSeek-V3 §2.2, eq. 21-22: one prediction depth."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_norm: RMSNorm.Config
        embedding_norm: RMSNorm.Config
        projection: Linear.Config
        block: Llama3TransformerBlock.Config

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_norm = config.hidden_norm.build()
        self.embedding_norm = config.embedding_norm.build()
        self.projection = config.projection.build()
        self.block = config.block.build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        shifted_embeddings: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        
        h_concat = torch.cat([self.hidden_norm(hidden_states), self.embedding_norm(shifted_embeddings)], dim=-1)

        h_proj = self.projection(h_concat)

        h_block = self.block(h_proj, attention_masks, positions)

        return h_block


class DeepSeekModel(Llama3Model):
    """DeepSeek-V3 §2.2: D sequential MTP modules on a shared trunk, shared
    embedding and unembedding, complete causal chain at each depth."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        mtp_modules: list

        def get_nparams_and_flops(
            self, model: Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            blocks = list(self.layers) + [
                mtp_module.block for mtp_module in self.mtp_modules
            ]
            for block in blocks:
                attention = block.attention
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.mtp_modules = ModuleDict()
        for index, mtp_module_config in enumerate(config.mtp_modules):
            self.mtp_modules[str(index)] = mtp_module_config.build()

    @property
    def num_predictions(self) -> int:
        return len(self.mtp_modules) + 1

    def trunk_hidden_states(
        self,
        token_embeddings: torch.Tensor,
        positions: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        h = token_embeddings
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)
        return h

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.norm(hidden_states)
        return normed if self._skip_lm_head else self.lm_head(normed)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> tuple[torch.Tensor, ...]:

        token_embeddings = self.tok_embeddings(tokens)

        hidden_states = self.trunk_hidden_states(
            token_embeddings, positions, attention_masks
        )

        outputs = [self.decode(hidden_states)]

        for i, mtp_module in enumerate(self.mtp_modules.values(), start=1):

            shifted_embeddings = torch.cat(
                [token_embeddings[i:],
                torch.zeros(i, token_embeddings.shape[-1], dtype=token_embeddings.dtype, device=token_embeddings.device)
            ])

            hidden_states = mtp_module(
                hidden_states, shifted_embeddings, attention_masks, positions
            )
            outputs.append(self.decode(hidden_states))

        return tuple(outputs)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions", None)
        padding_mask = batch.pop("padding_mask", None)

        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(inner, (FlexAttention.Config, VarlenAttention.Config)):
                batch["attention_masks"] = self.get_attention_masks(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        input_sharding = decoder_input_sharding()
        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        if parallelism.spmd_backend == "spmd_types":
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        base_labels = batch.pop("labels")
        positions = batch.get("positions", None)

        labels = mtp_labels(base_labels, positions, self.num_predictions)

        return inputs, labels, batch
