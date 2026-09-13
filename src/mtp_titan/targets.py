import torch

from torchtitan.components.loss import IGNORE_INDEX


def shifted_labels(
    base_labels: torch.Tensor,
    positions: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    if offset == 0:
        return base_labels

    shifted = torch.cat(
        [
            base_labels[offset:],
            torch.full(
                (offset,),
                IGNORE_INDEX,
                dtype=base_labels.dtype,
                device=base_labels.device,
            ),
        ]
    )

    segment_starts = torch.where(positions[1:] == 0)[0] + 1
    for start in segment_starts:
        shifted[max(int(start) - offset, 0) : start] = IGNORE_INDEX

    return shifted


def mtp_labels(
    base_labels: torch.Tensor,
    positions: torch.Tensor,
    num_predictions: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        shifted_labels(base_labels, positions, offset)
        for offset in range(num_predictions)
    )
