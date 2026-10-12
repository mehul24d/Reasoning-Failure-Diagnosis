from rfd.metrics.segmentation_metrics import boundary_f1, phase_recovery_rate, temporal_iou

PHASES = ["problem_understanding", "operation_planning", "arithmetic_execution", "self_verification"]


def _segments(*spans):
    return [{"start": s, "end": e, "phase": p} for s, e, p in spans]


def test_boundary_f1_perfect_match():
    gold = _segments((0, 5, "problem_understanding"), (5, 10, "arithmetic_execution"))
    pred = _segments((0, 5, "problem_understanding"), (5, 10, "arithmetic_execution"))
    result = boundary_f1(pred, gold, tolerance=0)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_boundary_f1_within_tolerance():
    gold = _segments((0, 5, "a"), (5, 10, "b"))
    pred = _segments((0, 6, "a"), (6, 10, "b"))  # boundary off by 1
    result = boundary_f1(pred, gold, tolerance=2)
    assert result["f1"] == 1.0


def test_boundary_f1_outside_tolerance():
    gold = _segments((0, 5, "a"), (5, 10, "b"))
    pred = _segments((0, 9, "a"), (9, 10, "b"))  # boundary off by 4
    result = boundary_f1(pred, gold, tolerance=1)
    assert result["f1"] == 0.0


def test_boundary_f1_no_boundaries_either_side():
    result = boundary_f1(_segments((0, 10, "a")), _segments((0, 10, "a")), tolerance=1)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_temporal_iou_identical():
    segs = _segments((0, 4, "problem_understanding"), (4, 10, "arithmetic_execution"))
    result = temporal_iou(segs, segs, PHASES)
    assert result["problem_understanding"] == 1.0
    assert result["arithmetic_execution"] == 1.0
    assert result["mean"] == 1.0  # only phases present in gold/pred count toward the mean


def test_temporal_iou_partial_overlap():
    gold = _segments((0, 10, "arithmetic_execution"))
    pred = _segments((5, 10, "arithmetic_execution"))
    result = temporal_iou(pred, gold, PHASES)
    assert result["arithmetic_execution"] == 0.5


def test_phase_recovery_rate_full_recovery():
    gold = _segments((0, 5, "problem_understanding"), (5, 10, "arithmetic_execution"))
    pred = _segments((0, 5, "problem_understanding"), (5, 10, "arithmetic_execution"))
    assert phase_recovery_rate(pred, gold, PHASES) == 1.0


def test_phase_recovery_rate_partial():
    gold = _segments((0, 5, "problem_understanding"), (5, 10, "arithmetic_execution"))
    pred = _segments((0, 5, "problem_understanding"))  # missing arithmetic_execution
    assert phase_recovery_rate(pred, gold, PHASES) == 0.5
