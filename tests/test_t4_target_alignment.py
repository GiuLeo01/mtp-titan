"""T4: target alignment. Head k reads the label stream k slots ahead, and
must fall back to IGNORE_INDEX once that slot leaves its own packed segment."""

from torchtitan.components.loss import IGNORE_INDEX

from .conftest import requires_cuda, NUM_SEGMENTS, SEGMENT_LENGTH


@requires_cuda
def test_target_alignment(build_gloeckle, packed_batch, parallel_dims, parallelism):
    num_heads = 3
    model = build_gloeckle(num_heads=num_heads)
    _inputs, labels, _extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    tokens = packed_batch["input"]
    base_labels = packed_batch["labels"]

    for head_index in range(num_heads):
        for segment in range(NUM_SEGMENTS):
            segment_start = segment * SEGMENT_LENGTH
            segment_end = segment_start + SEGMENT_LENGTH

            for pos in range(segment_start, segment_end):
                source = pos + head_index
                expected = (
                    base_labels[source] if source < segment_end else IGNORE_INDEX
                )
                assert labels[head_index][pos] == expected, (
                    f"head {head_index}, pos {pos}: expected {expected}, "
                    f"got {labels[head_index][pos]}"
                )

                if source + 1 < segment_end:
                    assert labels[head_index][pos] == tokens[source + 1]
