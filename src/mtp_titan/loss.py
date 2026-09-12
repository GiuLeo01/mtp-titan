from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch

from torchtitan.components.loss import BaseLoss, cross_entropy_loss
from torchtitan.config import CompileConfig
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend


class GloeckleLoss(BaseLoss):
    """Gloeckle et al., ICML 2024, §2, eq. 2: sum of the per-head cross entropies."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        global_vocab_size: int | None = None

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: tuple[torch.Tensor, ...],
        labels: tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        per_head_losses = []
        for p, l, in zip(pred, labels):
            head_loss = self.fn(p, l, global_vocab_size=self.global_vocab_size)
            per_head_losses.append(head_loss)
        
        loss = sum(per_head_losses)

        if get_spmd_backend() == "spmd_types" and current_spmd_mesh() is not None:
            spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})
            if global_valid_tokens is not None:
                spmd.assert_type(
                    global_valid_tokens,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss, {}

