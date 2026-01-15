#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit preload application (bolt surfaces + bolt deltas).

What it checks:
- Whether each bolt up/down surface resolves and samples distinct points.
- The inferred bolt axis from the up-surface normals (area-weighted mean normal).
- (Optional) Use a SavedModel to evaluate u(X; P, order) and compute bolt deltas Δ_i.

Usage examples:
  python tools/audit_preload.py --config config.yaml
  python tools/audit_preload.py --config config.yaml --saved-model auto --cases 2000,0,0 0,2000,0 0,0,2000
  python tools/audit_preload.py --saved-model results/saved_model_YYYYMMDD-HHMMSS --order 1,3,2 --cases 1500,1500,1500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return {} if data is None else dict(data)


def _resolve_surface_key(asm: Any, key: str) -> str:
    key = str(key or "").strip()
    if not key:
        raise KeyError("Empty surface key")
    surfaces = getattr(asm, "surfaces", {}) or {}
    if key in surfaces:
        return key
    low = key.lower()
    for k, s in surfaces.items():
        if low in str(k).lower():
            return str(k)
        try:
            if low == str(getattr(s, "name", "")).strip().lower():
                return str(k)
        except Exception:
            continue
    raise KeyError(f"Cannot resolve surface key '{key}' (available={len(surfaces)})")


def _parse_vec(text: str, *, dtype: Any, n: int) -> np.ndarray:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(parts)}: {text}")
    return np.asarray([dtype(p) for p in parts], dtype=np.float32 if dtype is float else np.int32)


def _find_latest_saved_model(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("saved_model_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class BoltAudit:
    name: str
    n_up: int
    n_dn: int
    centroid_up: np.ndarray
    centroid_dn: np.ndarray
    axis_up: np.ndarray


def _audit_preload_geometry(preload: Any) -> List[BoltAudit]:
    audits: List[BoltAudit] = []
    bolts = list(getattr(preload, "_bolts", []) or [])
    for b in bolts:
        X_up = np.asarray(getattr(b, "X_up"), dtype=np.float64).reshape(-1, 3)
        X_dn = np.asarray(getattr(b, "X_dn"), dtype=np.float64).reshape(-1, 3) if getattr(b, "X_dn") is not None else np.zeros((0, 3), dtype=np.float64)
        N_up = np.asarray(getattr(b, "N_up"), dtype=np.float64).reshape(-1, 3)
        w_up = np.asarray(getattr(b, "w_up"), dtype=np.float64).reshape(-1)
        w_sum = float(w_up.sum()) if w_up.size else 1.0
        axis = (N_up * w_up[:, None]).sum(axis=0) / (w_sum + 1e-16)
        axis /= float(np.linalg.norm(axis) + 1e-16)

        audits.append(
            BoltAudit(
                name=str(getattr(b, "name", "")),
                n_up=int(X_up.shape[0]),
                n_dn=int(X_dn.shape[0]),
                centroid_up=X_up.mean(axis=0) if X_up.size else np.zeros((3,), dtype=np.float64),
                centroid_dn=X_dn.mean(axis=0) if X_dn.size else np.zeros((3,), dtype=np.float64),
                axis_up=axis.astype(np.float64),
            )
        )
    return audits


def _print_geometry_report(audits: Sequence[BoltAudit]) -> None:
    print("== Preload surface audit ==")
    for i, a in enumerate(audits, start=1):
        cu = a.centroid_up
        cd = a.centroid_dn
        ax = a.axis_up
        print(
            f"[{i}] {a.name}: n_up={a.n_up} n_dn={a.n_dn} "
            f"centroid_up=({cu[0]:.3f},{cu[1]:.3f},{cu[2]:.3f}) "
            f"centroid_dn=({cd[0]:.3f},{cd[1]:.3f},{cd[2]:.3f}) "
            f"axis=({ax[0]:.3f},{ax[1]:.3f},{ax[2]:.3f})"
        )

    if len(audits) >= 2:
        print("== Pairwise centroid distances (up) ==")
        centers = np.stack([a.centroid_up for a in audits], axis=0)
        for i in range(len(audits)):
            for j in range(i + 1, len(audits)):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                print(f"{audits[i].name} <-> {audits[j].name}: {d:.3f}")


def _make_u_fn_from_saved_model(saved_model_dir: Path):
    import tensorflow as tf

    module = tf.saved_model.load(str(saved_model_dir))
    if not hasattr(module, "run"):
        raise AttributeError(f"SavedModel has no attribute 'run': {saved_model_dir}")

    def u_fn(X, params=None):
        params = {} if params is None else dict(params)
        P = params.get("P", None)
        if P is None:
            raise KeyError("params['P'] is required")
        order = params.get("order", None)
        if order is None:
            raise KeyError("params['order'] is required (length = n_bolts)")
        x = tf.convert_to_tensor(X, dtype=tf.float32)
        p = tf.reshape(tf.convert_to_tensor(P, dtype=tf.float32), (-1,))
        o = tf.reshape(tf.convert_to_tensor(order, dtype=tf.int32), (-1,))
        return module.run(x, p, o)

    return u_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument(
        "--saved-model",
        default=None,
        help="SavedModel directory, or 'auto' to pick the latest under results/",
    )
    ap.add_argument("--n-points-each", type=int, default=None, help="Override preload_n_points_each")
    ap.add_argument(
        "--order",
        default="1,2,3",
        help="Tightening order (1-based or 0-based), e.g. 1,3,2",
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Preload cases, e.g. 2000,0,0 0,2000,0 0,0,2000",
    )
    args = ap.parse_args()

    # Ensure repo modules are importable when running from tools/.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    inp_path = Path(str(cfg.get("inp_path") or "")).expanduser()
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")

    from inp_io.inp_parser import load_inp
    from physics.preload_model import BoltSurfaceSpec, PreloadConfig, PreloadWork

    asm = load_inp(str(inp_path))

    bolts_cfg = list(cfg.get("bolts", []) or [])
    if not bolts_cfg:
        raise ValueError("config has no 'bolts' list")

    specs: List[BoltSurfaceSpec] = []
    for b in bolts_cfg:
        name = str(b.get("name") or "")
        up = _resolve_surface_key(asm, str(b.get("up_surface_key") or b.get("up_key") or ""))
        dn = _resolve_surface_key(asm, str(b.get("down_surface_key") or b.get("down_key") or ""))
        specs.append(BoltSurfaceSpec(name=name, up_key=up, down_key=dn))

    n_points_each = int(args.n_points_each or cfg.get("preload_n_points_each") or 800)
    seed = int(cfg.get("seed") or 0)

    preload_cfg = PreloadConfig()
    preload = PreloadWork(preload_cfg)
    preload.build_from_specs(asm, specs, n_points_each=n_points_each, seed=seed)

    audits = _audit_preload_geometry(preload)
    _print_geometry_report(audits)

    if args.saved_model is None and not args.cases:
        return 0

    saved_model_dir: Optional[Path] = None
    if args.saved_model:
        if str(args.saved_model).strip().lower() == "auto":
            saved_model_dir = _find_latest_saved_model(Path("results"))
            if saved_model_dir is None:
                raise FileNotFoundError("No SavedModel found under results/ (expected saved_model_*)")
        else:
            saved_model_dir = Path(args.saved_model)
        if saved_model_dir is not None and not saved_model_dir.exists():
            raise FileNotFoundError(f"SavedModel dir not found: {saved_model_dir}")

    if saved_model_dir is None:
        raise ValueError("--saved-model is required to evaluate cases")

    u_fn = _make_u_fn_from_saved_model(saved_model_dir)

    nb = len(specs)
    order = _parse_vec(args.order, dtype=int, n=nb).reshape(-1)
    if int(order.min()) >= 1 and int(order.max()) <= nb:
        order = order - 1

    if args.cases:
        cases = [_parse_vec(s, dtype=float, n=nb).reshape(-1) for s in args.cases]
    else:
        cases = [
            np.array([2000.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 2000.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 2000.0], dtype=np.float32),
        ]

    print(f"== SavedModel preload-delta check == ({saved_model_dir})")
    import tensorflow as tf

    for P in cases:
        params = {"P": tf.convert_to_tensor(P, dtype=tf.float32), "order": tf.convert_to_tensor(order, dtype=tf.int32)}
        W_pre, stats = preload.energy(u_fn, params)
        bd = None
        try:
            bd = stats.get("preload", {}).get("bolt_deltas")
        except Exception:
            bd = None
        bd_np = bd.numpy() if hasattr(bd, "numpy") else np.asarray(bd) if bd is not None else None
        W_val = float(W_pre.numpy()) if hasattr(W_pre, "numpy") else float(W_pre)
        p_txt = ",".join(str(int(x)) if float(x).is_integer() else f"{float(x):g}" for x in P.tolist())
        o_txt = ",".join(str(int(x) + 1) for x in order.tolist())
        if bd_np is None:
            print(f"P=[{p_txt}] order=[{o_txt}]  W_pre={W_val:.6e}  deltas=?")
        else:
            d_txt = ",".join(f"{float(x):.6e}" for x in bd_np.tolist())
            print(f"P=[{p_txt}] order=[{o_txt}]  W_pre={W_val:.6e}  deltas=[{d_txt}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
