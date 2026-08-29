"""Quantitative receipts for the actual worst-leaf quadtree refinement benchmark.

`benchmark_report.bench_quality_quantiles` returns base and post-refinement error
quantiles.  This adapter intentionally does not call that 'observation
preservation': it records which declared consumers improved, stayed within a
predeclared tolerance, or worsened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuantileCriterion:
    name: str
    max_final: dict[str, float]
    declared_before_evaluation: bool
    justification: str


@dataclass(frozen=True)
class QuantitativeRefinementReceipt:
    sparsity: float
    leaves_after_refinement: int
    base_quantiles: dict[str, float]
    final_quantiles: dict[str, float]
    deltas: dict[str, float]
    improved: dict[str, bool]
    criterion: QuantileCriterion | None
    criterion_satisfied: bool
    exact_observation_preservation_claimed: bool = False
    every_metric_monotone_claimed: bool = False
    physical_truth_claimed: bool = False

    @property
    def eligible_approximation_evidence(self) -> bool:
        return bool(
            self.criterion is not None
            and self.criterion.declared_before_evaluation
            and self.criterion_satisfied
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["eligible_approximation_evidence"] = self.eligible_approximation_evidence
        return result


def receipt_from_quality_quantile_row(
    row: tuple[float, int, float, float, float, float, float, float, float, float],
    *,
    criterion: QuantileCriterion | None = None,
) -> QuantitativeRefinementReceipt:
    (
        sparsity,
        leaves,
        q25,
        q50,
        q75,
        q99,
        q25r,
        q50r,
        q75r,
        q99r,
    ) = row

    base = {"q25": q25, "q50": q50, "q75": q75, "q99": q99}
    final = {"q25": q25r, "q50": q50r, "q75": q75r, "q99": q99r}
    deltas = {key: final[key] - base[key] for key in base}
    improved = {key: final[key] <= base[key] for key in base}

    criterion_satisfied = False
    if criterion is not None and criterion.declared_before_evaluation:
        criterion_satisfied = all(
            key in final and final[key] <= float(limit)
            for key, limit in criterion.max_final.items()
        )

    return QuantitativeRefinementReceipt(
        sparsity=float(sparsity),
        leaves_after_refinement=int(leaves),
        base_quantiles={key: float(value) for key, value in base.items()},
        final_quantiles={key: float(value) for key, value in final.items()},
        deltas={key: float(value) for key, value in deltas.items()},
        improved=improved,
        criterion=criterion,
        criterion_satisfied=criterion_satisfied,
    )
