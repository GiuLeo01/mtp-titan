from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.protocols.module import Module, ModuleDict


class GloeckleModel(Llama3Model):
    """Gloeckle et al., ICML 2024, §3: shared trunk, n independent one-block heads,
    shared unembedding."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        heads: list

        def get_nparams_and_flops(
            self, model: Module, seq_len: int
        ) -> tuple[int, int]:
            raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
