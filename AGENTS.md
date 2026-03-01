# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python scripts drive experiments and demos (e.g., `metric_lyapunov_quadtree_demo.py`, `pda_mdl_light_transport_test.py`).
- Generated artifacts and reference visuals are stored as images in the repo root and in `outputs_pda/`.
- Research notes and references live as PDFs in the repo root.

If you add new code, prefer keeping core logic in a dedicated module file and keep demo scripts thin (imports + parameters + plotting).

## Build, Test, and Development Commands
This repo is script-first; there is no build system or package config yet.
- `python metric_lyapunov_quadtree_demo.py` — runs the metric/lyapunov/quadtree demo and emits figures.
- `python pda_mdl_light_transport_test.py` — runs the PDA/MDL light transport test pipeline.
- `python pda_mdl_light_transport_test_fixed.py` — fixed variant of the above.
- `python dashi_test_pda_mdl_light_transport_kernel.py` — kernel-focused test run.

If you add dependencies, document them in a `requirements.txt` and keep them minimal.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 naming: `snake_case` for functions/vars and `PascalCase` for classes.
- Keep functions small and pure where possible; isolate plotting from computation.
- Prefer explicit names over abbreviations (e.g., `transport_kernel` over `tk`).

## Testing Guidelines
There is no formal test framework yet. Test scripts are conventionally named `*_test.py` and run directly with `python`.
- If you add tests, keep them deterministic and save outputs with clear filenames in `outputs_pda/`.
- Validate numerics with simple assertions before plotting.

## Commit & Pull Request Guidelines
Git history is minimal (`init`), so there is no established convention. Use short, imperative commit messages (e.g., "Add metric regularizer").
For PRs:
- Include a concise summary of changes.
- Attach before/after images when outputs change.
- Link any relevant issues or notes.

## Data & Artifacts
Large images and generated plots should live in `outputs_pda/` or use descriptive filenames in the repo root. Avoid committing temporary files or redundant exports.
