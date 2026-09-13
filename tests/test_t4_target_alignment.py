"""T4: target alignment. Prediction k reads the label stream k slots ahead, and
must fall back to IGNORE_INDEX once that slot leaves its own packed segment."""

from torchtitan.components.loss import IGNORE_INDEX

from .conftest import requires_cuda, NUM_SEGMENTS, SEGMENT_LENGTH


@requires_cuda
def test_target_alignment(build_mtp, packed_batch, parallel_dims, parallelism):
    num_predictions = 3
    model = build_mtp(num_predictions)
    _inputs, labels, _extra_kwargs = model.preprocess_inputs(
        dict(packed_batch), parallel_dims=parallel_dims, parallelism=parallelism
    )
    tokens = packed_batch["input"]
    base_labels = packed_batch["labels"]

    assert len(labels) == num_predictions

    for prediction in range(num_predictions):
        for segment in range(NUM_SEGMENTS):
            segment_start = segment * SEGMENT_LENGTH
            segment_end = segment_start + SEGMENT_LENGTH

            for pos in range(segment_start, segment_end):
                source = pos + prediction
                expected = (
                    base_labels[source] if source < segment_end else IGNORE_INDEX
                )
                assert labels[prediction][pos] == expected, (
                    f"prediction {prediction}, pos {pos}: expected {expected}, "
                    f"got {labels[prediction][pos]}"
                )

                if source + 1 < segment_end:
                    assert labels[prediction][pos] == tokens[source + 1]
