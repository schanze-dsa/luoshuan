# Repository Guidelines

## Project Structure & Module Organization
- Entry point: `main.py` (loads `config.yaml`, builds DFEM + contact + preload, trains, optionally exports a SavedModel).
- Key inputs: `config.yaml` (materials/weights/sampling), `shuangfan.inp` (Abaqus INP), optional meshes like `mirror_up_simple.ply`.
- Core code is under `src/`: `assembly/` (surface helpers), `inp_io/` (INP parser), `mesh/` (triangulation/contact sampling), `model/` (network + total energy), `physics/` (DFEM/contact/preload), `train/` (trainer + adaptive loss weights), `viz/` (postprocessing).
- Outputs: `results/` (plots/data) and `checkpoints/run-YYYYMMDD-HHMMSS/` (per-run checkpoints).

## Build, Test, and Development Commands
- Train: `python main.py`.
- Export SavedModel: `python main.py --export results/saved_model_my_run`.
- Quick checks: `python test_config_read.py`, `python test_dfem.py`.
- Optional search: `python optuna_search.py` (edit parameters inside the file).

## Coding Style & Naming Conventions
- PEP 8, 4-space indents. Use type hints and small helpers; prefer dataclasses for configs.
- Naming: `snake_case` for funcs/vars, `UpperCamelCase` for classes, config keys `lower_snake_case`.
- Keep training logs short and stable (avoid per-sample spam).

## Testing Guidelines
- Prefer lightweight `test_*.py` smoke tests (shape/forward-pass assertions). Avoid long training loops.
- Seed randomness when tests sample (`contact_seed`, NumPy/TF seeds).

## Commit & Pull Request Guidelines
- Commits: short imperative subject (e.g., `train: stabilize contact ALM`), add a brief body for non-obvious changes.
- PRs: describe what/why, list config keys touched, include commands run, and attach representative plots/paths when outputs change.

## Configuration & Data Notes
- Keep TF env vars near the top of `main.py` (before importing TensorFlow).
- Prefer changing `config.yaml` over hard-coded constants; keep units explicit and self-consistent.
- Don’t commit large artifacts (`checkpoints/`, `results/`); clean them before publishing.

## PINN/DFEM Design Notes
- The displacement field supports coordinate/Fourier features or DFEM node embeddings (`dfem_mode`). The GCN uses a kNN graph built from mesh nodes.
- Stress supervision uses a stress head (`stress_out_dim=6`) and a scalar weight (`stress_loss_weight`); set `yield_strength` to log `σvm/σy`.
