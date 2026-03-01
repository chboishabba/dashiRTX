# ROADMAP

## Current State (Implemented)
- Metric‑regularized MDL loss, ternary carrier penalty, and contraction‑aware options in `pda_mdl_light_transport_test.py`.
- Two‑stage diffusion → projection training mode.
- Kernel‑operator learning stub (K‑based model).
- Quadtree ultrametric renderer with vectorized + numba paths.
- Bounce‑cost modeling to simulate multi‑bounce kernel expense.
- Benchmarks: sparsity sweeps, size sweeps (up to 2048), and quality quantiles (q25/q50/q75/q99).
- Optional refinement loop for quadtree quality and an error‑boost refinement loop in training (`--refine_training`).

## Near‑Term (Next 1–2 Iterations)
- Add quality‑targeted refinement (stop when q99 ≤ target or leaves ≥ min).
- Make multi‑frame training TV/metric loss well‑defined (batched or per‑frame aggregation).
- Integrate refinement into the rendering/training loop (render → error → refine → retrain).
- Improve sparsity controls to avoid leaf collapse while preserving speed.

## Longer‑Term
- GPU kernel for block traversal + rendering.
- Operator learning integrated into the main pipeline (K(x, view) as primary invariant).
- Metric‑aware contraction diagnostics in training visualizations.
