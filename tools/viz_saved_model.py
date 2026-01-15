#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mirror deflection plots from an exported TensorFlow SavedModel (no retraining).

Why this exists
---------------
Training runs can take a long time. When you only need to regenerate/verify the
visualization (e.g. after changing `src/viz/mirror_viz.py`), you can do it from
the exported SavedModel directly.

Examples
--------
  python tools/viz_saved_model.py --config config.yaml --saved-model auto
  python tools/viz_saved_model.py --saved-model results/saved_model_YYYYMMDD-HHMMSS --cases 2000,0,0
  python tools/viz_saved_model.py --saved-model auto --cases 1500,1500,1500 --order 3,2,1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return {} if data is None else dict(data)


def _find_latest_saved_model(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("saved_model_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_vec(text: str, *, dtype: Any, n: int) -> np.ndarray:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(parts)}: {text}")
    return np.asarray([dtype(p) for p in parts], dtype=np.float32 if dtype is float else np.int32)


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    p = Path(str(raw or "")).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_saved_u_fn(saved_model_dir: Path):
    import tensorflow as tf

    loaded = tf.saved_model.load(str(saved_model_dir))
    serving = loaded.signatures.get("serving_default")
    if serving is None:
        keys = sorted(list(loaded.signatures.keys()))
        raise KeyError(f"SavedModel has no 'serving_default' signature (available={keys})")

    def u_fn(X, params=None):
        P = None
        order = None
        if isinstance(params, dict):
            P = params.get("P")
            order = params.get("order")
            if order is None:
                order = params.get("stage_order")
        if P is None:
            P = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        if order is None:
            order = tf.constant([1, 2, 3], dtype=tf.int32)

        out = serving(
            x=tf.convert_to_tensor(X, tf.float32),
            p=tf.reshape(tf.convert_to_tensor(P, tf.float32), (-1,)),
            order=tf.reshape(tf.convert_to_tensor(order, tf.int32), (-1,)),
        )
        return out["output_0"]

    return u_fn


def _default_cases(cfg: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    preload_range = cfg.get("preload_range_n", [0.0, 2000.0])
    try:
        max_p = float(preload_range[1])
    except Exception:
        max_p = 2000.0
    equal_p = 0.75 * max_p

    cases: List[Tuple[np.ndarray, np.ndarray, str]] = []
    # 3 single-bolt cases, fixed order 1-2-3
    for P in (
        [max_p, 0.0, 0.0],
        [0.0, max_p, 0.0],
        [0.0, 0.0, max_p],
    ):
        cases.append((np.asarray(P, np.float32), np.asarray([1, 2, 3], np.int32), "123"))

    # 3 equal-preload cases with different tightening orders
    for order in ([1, 2, 3], [1, 3, 2], [3, 2, 1]):
        cases.append((np.asarray([equal_p, equal_p, equal_p], np.float32), np.asarray(order, np.int32), "".join(str(x) for x in order)))

    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot mirror deflection from a SavedModel.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--saved-model",
        default="auto",
        help="SavedModel directory, or 'auto' to pick the latest under results/",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for PNG/TXT (default: output_config.save_path or ./results)",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Preload cases, e.g. 2000,0,0 0,2000,0 0,0,2000 (default: 6 fixed cases)",
    )
    parser.add_argument(
        "--order",
        default="",
        help="Tightening order (1-based or 0-based), e.g. 1,3,2 (applies to all --cases)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    from inp_io.inp_parser import load_inp
    from viz.mirror_viz import plot_mirror_deflection_by_name

    out_cfg = cfg.get("output_config", {}) or {}
    out_dir = str(args.out_dir or out_cfg.get("save_path") or "./results")
    out_dir_path = _resolve_path(out_dir, base_dir=cfg_path.parent)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    inp_path = _resolve_path(str(cfg.get("inp_path") or "shuangfan.inp"), base_dir=cfg_path.parent)
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")
    asm = load_inp(str(inp_path))

    saved_arg = str(args.saved_model or "").strip()
    saved_dir: Optional[Path]
    if saved_arg.lower() == "auto":
        saved_dir = _find_latest_saved_model(out_dir_path)
        if saved_dir is None:
            saved_dir = _find_latest_saved_model(repo_root / "results")
    else:
        saved_dir = _resolve_path(saved_arg, base_dir=cfg_path.parent)
    if saved_dir is None or not saved_dir.exists():
        raise FileNotFoundError(f"SavedModel not found: {saved_dir}")

    u_fn = _load_saved_u_fn(saved_dir)

    mirror_name = str(cfg.get("mirror_surface_name") or "MIRROR up")

    viz_surface_source = str(out_cfg.get("viz_surface_source", "surface"))
    viz_refine_subdivisions = int(out_cfg.get("viz_refine_subdivisions", 2))
    viz_refine_max_points = out_cfg.get("viz_refine_max_points", None)
    viz_style = str(out_cfg.get("viz_style", "smooth"))
    viz_colormap = str(out_cfg.get("viz_colormap", "turbo"))
    viz_levels = int(out_cfg.get("viz_levels", 64))
    viz_units = str(out_cfg.get("viz_units", "mm"))
    viz_draw_wireframe = bool(out_cfg.get("viz_draw_wireframe", False))
    viz_eval_batch_size = int(out_cfg.get("viz_eval_batch_size", 65_536))
    viz_eval_scope = str(out_cfg.get("viz_eval_scope", "assembly"))
    viz_use_shape_interp = bool(out_cfg.get("viz_use_shape_function_interp", False))
    viz_diagnose_blanks = bool(out_cfg.get("viz_diagnose_blanks", False))
    viz_auto_fill_blanks = bool(out_cfg.get("viz_auto_fill_blanks", False))
    viz_remove_rigid = bool(out_cfg.get("viz_remove_rigid", True))
    write_data = bool(out_cfg.get("viz_write_data", True))
    write_surface_mesh = bool(out_cfg.get("viz_write_surface_mesh", False))

    order_override = None
    if args.order.strip():
        order_override = _parse_vec(args.order, dtype=int, n=3).astype(np.int32)

    cases: List[Tuple[np.ndarray, np.ndarray, str]] = []
    if args.cases:
        for entry in args.cases:
            P = _parse_vec(entry, dtype=float, n=3)
            order = np.asarray([1, 2, 3], dtype=np.int32) if order_override is None else order_override
            tag = "".join(str(int(x)) for x in order.tolist())
            cases.append((P.astype(np.float32), order.astype(np.int32), tag))
    else:
        cases = _default_cases(cfg)

    title_prefix = str(out_cfg.get("viz_title_prefix", "Total Deformation (SavedModel)"))

    for i, (P, order, tag) in enumerate(cases, start=1):
        suffix = f"_{tag}"
        png_path = out_dir_path / f"deflection_{i:02d}{suffix}.png"
        txt_path = out_dir_path / f"deflection_{i:02d}{suffix}.txt" if write_data else None
        mesh_out = "auto" if write_surface_mesh else None
        order_display = "-".join(str(int(x)) for x in order.tolist())
        title = f"{title_prefix}  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N  (order={order_display})"
        params = {"P": P, "order": order}
        diag_out = {} if viz_diagnose_blanks else None

        plot_mirror_deflection_by_name(
            asm,
            mirror_name,
            u_fn=u_fn,
            params=params,
            P_values=tuple(float(x) for x in P.reshape(-1)),
            out_path=str(png_path),
            render_surface=True,
            surface_source=viz_surface_source,
            title_prefix=title,
            units=viz_units,
            levels=viz_levels,
            style=viz_style,
            cmap=viz_colormap,
            draw_wireframe=viz_draw_wireframe,
            refine_subdivisions=viz_refine_subdivisions,
            refine_max_points=None if viz_refine_max_points is None else int(viz_refine_max_points),
            use_shape_function_interp=viz_use_shape_interp,
            data_out_path=None if txt_path is None else str(txt_path),
            surface_mesh_out_path=mesh_out,
            eval_batch_size=viz_eval_batch_size,
            eval_scope=viz_eval_scope,
            diagnose_blanks=viz_diagnose_blanks,
            auto_fill_blanks=viz_auto_fill_blanks,
            remove_rigid=viz_remove_rigid,
            diag_out=diag_out,
            show=False,
        )

        print(f"[viz_saved_model] saved -> {png_path}")
        if txt_path is not None:
            print(f"[viz_saved_model] data  -> {txt_path}")
        if diag_out and diag_out.get("blank_check") is not None:
            blank = diag_out["blank_check"]
            print(f"[viz_saved_model] blank_check: {blank.primary_cause}")

    print(f"[viz_saved_model] SavedModel used: {saved_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

