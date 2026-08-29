"""BIDI receipt adapter for dashiRTX light-transport experiments.

This does not change rendering or learning numerics.  It packages benchmark
outputs so held-out approximation quality, compression, and runtime remain
separate claims, and requires criterion provenance before an approximation
claim can be promoted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HeldOutRenderCriterion:
    max_mae: float
    declared_before_heldout_evaluation: bool
    justification: str


@dataclass(frozen=True)
class LightTransportReceipt:
    reference_kind: str
    approximation_kind: str
    heldout_mae: float
    leaf_count: int | None
    reference_seconds: float | None
    approximation_seconds: float | None
    criterion: HeldOutRenderCriterion | None
    finite_run_only: bool = True
    lower_mae_is_physical_truth: bool = False
    fewer_leaves_is_observation_preservation: bool = False
    faster_render_is_semantic_equivalence: bool = False

    @property
    def criterion_predeclared(self) -> bool:
        return bool(
            self.criterion is not None
            and self.criterion.declared_before_heldout_evaluation
        )

    @property
    def heldout_approximation_supported(self) -> bool:
        return bool(
            self.criterion_predeclared
            and self.heldout_mae <= self.criterion.max_mae
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["criterion_predeclared"] = self.criterion_predeclared
        data["heldout_approximation_supported"] = self.heldout_approximation_supported
        return data


def receipt_from_quadtree_benchmark(
    benchmark: dict[str, Any],
    *,
    approximation_key: str = "quadtree",
    criterion: HeldOutRenderCriterion | None = None,
) -> LightTransportReceipt:
    """Package one pixel-vs-quadtree benchmark result.

    Existing ``benchmark_report.bench_quadtree`` output already carries the
    reference runtime plus approximation runtime/leaves/MAE.  No threshold is
    invented here: without an externally supplied predeclared criterion the
    result remains finite empirical evidence only.
    """

    reference = benchmark["pixel"]
    approximation = benchmark[approximation_key]
    return LightTransportReceipt(
        reference_kind="pixel",
        approximation_kind=approximation_key,
        heldout_mae=float(approximation["mae"]),
        leaf_count=(
            int(approximation["leaves"])
            if approximation.get("leaves") is not None
            else None
        ),
        reference_seconds=(
            float(reference["sec"])
            if reference.get("sec") is not None
            else None
        ),
        approximation_seconds=(
            float(approximation["sec"])
            if approximation.get("sec") is not None
            else None
        ),
        criterion=criterion,
    )
