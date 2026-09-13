from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module, ModuleDict

from .targets import mtp_labels


class GloeckleModel(Llama3Model):
    """Gloeckle et al., ICML 2024, §3: shared trunk, n independent one-block heads,
    shared unembedding."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        heads: list

        def get_nparams_and_flops(
            self, model: Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for block in list(self.layers) + list(self.heads):
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
        self.heads = ModuleDict()
        for index, head_config in enumerate(config.heads):
            self.heads[str(index)] = head_config.build()

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    def trunk_hidden_states(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        # based on the common decoder class, but without the lm_head, which will be applied in the mtp heads 

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        return h


    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> tuple[torch.Tensor, ...]:
        
        h_trunk = self.trunk_hidden_states(tokens, positions, attention_masks)

        outputs = []
        for mtp_head in self.heads.values():
            h_head = mtp_head(h_trunk, attention_masks, positions)
            h_head = self.norm(h_head)
            if self._skip_lm_head:
                outputs.append(h_head)
            else:
                outputs.append(self.lm_head(h_head))

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

        labels = mtp_labels(base_labels, positions, self.num_heads)

        return inputs, labels, batch
