from refinement_receipt import QuantileCriterion, receipt_from_quality_quantile_row


def test_refinement_receipt_keeps_mixed_metric_direction() -> None:
    row = (
        0.25,
        16,
        0.07212,
        0.14190,
        0.21150,
        0.52460,
        0.07060,
        0.14032,
        0.21214,
        0.48160,
    )
    receipt = receipt_from_quality_quantile_row(row)
    assert receipt.improved["q25"] is True
    assert receipt.improved["q50"] is True
    assert receipt.improved["q75"] is False
    assert receipt.improved["q99"] is True
    assert receipt.exact_observation_preservation_claimed is False
    assert receipt.every_metric_monotone_claimed is False


def test_predeclared_consumer_tolerance_can_support_approximation() -> None:
    row = (0.25, 16, 0.08, 0.15, 0.22, 0.53, 0.07, 0.14, 0.21, 0.48)
    criterion = QuantileCriterion(
        name="heldout_render_budget",
        max_final={"q50": 0.15, "q99": 0.50},
        declared_before_evaluation=True,
        justification="consumer-owned display error budget",
    )
    receipt = receipt_from_quality_quantile_row(row, criterion=criterion)
    assert receipt.criterion_satisfied is True
    assert receipt.eligible_approximation_evidence is True


def test_posthoc_tolerance_is_not_eligible() -> None:
    row = (0.25, 16, 0.08, 0.15, 0.22, 0.53, 0.07, 0.14, 0.21, 0.48)
    criterion = QuantileCriterion(
        name="posthoc",
        max_final={"q99": 1.0},
        declared_before_evaluation=False,
        justification="seen after run",
    )
    receipt = receipt_from_quality_quantile_row(row, criterion=criterion)
    assert receipt.criterion_satisfied is False
    assert receipt.eligible_approximation_evidence is False
