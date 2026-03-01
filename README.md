# Repository Guidelines

This repository collects experiments and reference materials around the DASHI / PDA–MDL light‑transport formalism. It includes standalone Python demos, generated figures, and a set of PDFs that capture the mathematical framing (ternary carriers, kernel towers, MDL selection, and geometry/metric bridges).

## Project Structure
- Root-level scripts are the main entry points for experiments.
- Generated figures live in the repo root and in `outputs_pda/`.
- Reference writeups live as PDFs in the repo root.

## Quickstart
All scripts are self-contained and use `numpy`. Some plots require `matplotlib`.

Example runs:
- `python pda_mdl_light_transport_test.py --outdir outputs_pda --show`
- `python pda_mdl_light_transport_test_fixed.py --outdir outputs_pda --show`
- `python metric_lyapunov_quadtree_demo.py --show`
- `python quadtree_ultrametric_renderer.py`

## Key Scripts
- `pda_mdl_light_transport_test.py`: end‑to‑end PDA/MDL light‑transport toy with reprojection error, signed frontier `S0`, importance `I0`, and learned refresh mask.
- `pda_mdl_light_transport_test_fixed.py`: corrected variant of the above.
- `dashi_test_pda_mdl_light_transport_kernel.py`: kernel-focused test run.
- `metric_lyapunov_quadtree_demo.py`: metric‑aware Lyapunov descent + quadtree accumulation prototype.
- `quadtree_ultrametric_renderer.py`: minimal ultrametric quadtree renderer demo.

## Outputs & Figures
- `outputs_pda/` stores canonical figures (fields, masks, MDL curves, error reduction).
- Additional rendered outputs and figures are in the repo root (e.g., `lt_Figure_*.png`, `render_mdl_light_transport_Figure_*.png`).

## Benchmarks (Latest)
All numbers below are from `benchmark_report.py` with `bounce_cost=30` (simulated expensive multi‑bounce kernel).

Rendering size sweep (seconds, lower is better):
- N=512: pixel 0.0716, quadtree_numba 0.0850
- N=1024: pixel 0.3086, quadtree_numba 0.2348
- N=2048: pixel 1.2103, quadtree_numba 0.8978

Sparsity sweep (N=96; quadtree leaves + MAE vs pixel):
- s=1.00: leaves 205, MAE 0.1356
- s=0.50: leaves 184, MAE 0.1362
- s=0.25: leaves 1, MAE 0.1492
- s=0.10: leaves 1, MAE 0.1492
- s=0.05: leaves 1, MAE 0.1492
- s=0.02: leaves 1, MAE 0.1492

Quality quantiles (absolute error vs pixel, N=1024, refinement: 2 steps / top‑4 leaves):
- s=1.00: q25 0.06711 → 0.06711, q50 0.13430 → 0.13429, q75 0.20137 → 0.20137, q99 0.26830 → 0.26830
- s=0.50: q25 0.06762 → 0.06762, q50 0.13501 → 0.13500, q75 0.20236 → 0.20235, q99 0.27778 → 0.27779
- s=0.25+: q25 0.07212 → 0.07060, q50 0.14190 → 0.14032, q75 0.21150 → 0.21214, q99 0.52460 → 0.48160 (leaf collapse mitigated)

Run:
`python benchmark_report.py`

## Reference PDFs
- `DASHI Physics.pdf`: high‑level formalism notes and mappings to admissibility/MDL concepts.
- `DASHI - Branch · Visualising Collapse and Sparsity - RTX - light transport.pdf`: p‑adic/ternary kernel tower formalization and structured codec framing.
- `Branch · Formalism Bridging GR and MDL.pdf`: dictionary between GR/differential‑geometry notions and DASHI/MDL invariants.
- `Branch · Rubik's Cube and S(3).pdf`: group‑theoretic notes on ternary carriers and closure.
- `Branch · Engine Sound Simulation Methods.pdf`: additional domain notes (not currently wired to code).

## Notes
There is no package/build system yet. Keep new scripts self‑contained, prefer deterministic outputs, and write new figures into `outputs_pda/` or with descriptive filenames in the root.
