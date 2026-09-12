import os
from dataclasses import dataclass
from typing import Any

from torch.distributed._functional_collectives import all_reduce

from torchtitan.components.metrics import MetricsProcessor

from .loss import GloeckleLoss, TRAIN_HEAD_LOSSES, VALIDATION_HEAD_LOSSES


class MtpMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        run_name: str | None = None

    def __init__(self, config: Config, **kwargs):
        if config.run_name is not None:
            os.environ.setdefault("WANDB_RUN_NAME", config.run_name)
        super().__init__(config, **kwargs)

    def _per_head_cross_entropy(
        self, accumulator_key: str, prefix: str
    ) -> dict[str, float]:
        accumulated = GloeckleLoss.drain_head_losses(accumulator_key)
        if accumulated is None:
            return {}
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            accumulated = all_reduce(accumulated, reduceOp="sum", group=loss_mesh)
        loss_sums, valid_tokens = accumulated
        return {
            f"{prefix}/head_{index + 1}/cross_entropy": float(
                loss_sums[index] / valid_tokens[index]
            )
            for index in range(loss_sums.numel())
        }

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        super().log(
            step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics={
                **(extra_metrics or {}),
                **self._per_head_cross_entropy(TRAIN_HEAD_LOSSES, "loss_metrics"),
            },
        )

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ):
        super().log_validation(
            loss,
            step,
            extra_metrics={
                **(extra_metrics or {}),
                **self._per_head_cross_entropy(
                    VALIDATION_HEAD_LOSSES, "validation_metrics"
                ),
            },
        )
