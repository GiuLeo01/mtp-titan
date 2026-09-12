from dataclasses import dataclass
from typing import Any, ClassVar

import spmd_types as spmd
import torch

from torchtitan.components.loss import BaseLoss, cross_entropy_loss, IGNORE_INDEX
from torchtitan.config import CompileConfig
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend

TRAIN_HEAD_LOSSES = "train"
VALIDATION_HEAD_LOSSES = "validation"


class GloeckleLoss(BaseLoss):
    """Gloeckle et al., ICML 2024, §2, eq. 2: sum of the per-head cross entropies."""

    head_loss_accumulators: ClassVar[dict[str, torch.Tensor]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        global_vocab_size: int | None = None

    @classmethod
    def drain_head_losses(cls, accumulator_key: str) -> torch.Tensor | None:
        accumulator = cls.head_loss_accumulators.get(accumulator_key)
        if accumulator is None:
            return None
        drained = accumulator.clone()
        accumulator.zero_()
        return drained

    @classmethod
    def _accumulate_head_losses(
        cls,
        per_head_losses: list[torch.Tensor],
        labels: tuple[torch.Tensor, ...],
    ) -> None:
        accumulator_key = (
            TRAIN_HEAD_LOSSES if torch.is_grad_enabled() else VALIDATION_HEAD_LOSSES
        )
        accumulator = cls.head_loss_accumulators.get(accumulator_key)
        if accumulator is None:
            accumulator = torch.zeros(
                (2, len(per_head_losses)),
                dtype=torch.float32,
                device=per_head_losses[0].device,
            )
            cls.head_loss_accumulators[accumulator_key] = accumulator
        with spmd.no_typecheck(), torch.no_grad():
            observation = torch.stack(
                [
                    torch.stack([head_loss.detach() for head_loss in per_head_losses]),
                    torch.stack(
                        [(head_labels != IGNORE_INDEX).sum() for head_labels in labels]
                    ),
                ]
            ).float()
            accumulator.add_(observation)

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

        self._accumulate_head_losses(per_head_losses, labels)

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

