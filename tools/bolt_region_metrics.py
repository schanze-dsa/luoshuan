#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-bolt "near-hole" displacement metrics on the mirror surface.

Motivation
----------
Global argmax(|u|) on the mirror can stay at the same location across load cases
even when the fields differ. For bolt preload sanity checks, it's more useful to
measure |u| (or components) in neighborhoods around each bolt.

This script:
- Loads bolt up-surface samples from INP via PreloadWork and estimates each bolt
  center (centroid of sampled up points).
- Loads mirror deflection TXT exports (from `src/viz/mirror_viz.py`).
- For each bolt, selects mirror nodes within a given XY radius of the bolt center
  and reports max/mean/percentile |u| and the local hotspot UV.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    raise KeyError(f"Cannot resolve surface key '{key}'")


@dataclass(frozen=True)
class DeflectionData:
    path: Path
    xyz: np.ndarray  # (N,3)
    u: np.ndarray  # (N,3)
    umag: np.ndarray  # (N,)
    uv: np.ndarray  # (N,2)


def _load_deflection_txt(path: Path) -> DeflectionData:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            # node_id unused for region metrics
            rows.append([float(x) for x in parts[1:10]])

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    data = np.asarray(rows, dtype=np.float64)
    xyz = data[:, 0:3]
    u = data[:, 3:6]
    umag = data[:, 6]
    uv = data[:, 7:9]
    return DeflectionData(path=path, xyz=xyz, u=u, umag=umag, uv=uv)


@dataclass(frozen=True)
class BoltCenter:
    name: str
    center_xyz: np.ndarray  # (3,)


def _load_bolt_centers(cfg: Dict[str, Any]) -> List[BoltCenter]:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    from inp_io.inp_parser import load_inp
    from physics.preload_model import BoltSurfaceSpec, PreloadConfig, PreloadWork

    inp_path = Path(str(cfg.get("inp_path") or "")).expanduser()
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")
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

    n_points_each = int(cfg.get("preload_n_points_each") or 800)
    seed = int(cfg.get("seed") or 0)

    preload = PreloadWork(PreloadConfig())
    preload.build_from_specs(asm, specs, n_points_each=n_points_each, seed=seed)

    centers: List[BoltCenter] = []
    for b in getattr(preload, "_bolts", []) or []:
        X_up = np.asarray(getattr(b, "X_up"), dtype=np.float64).reshape(-1, 3)
        if X_up.size == 0:
            continue
        centers.append(BoltCenter(name=str(getattr(b, "name", "")), center_xyz=X_up.mean(axis=0)))
    return centers


def _region_stats(d: DeflectionData, center: np.ndarray, radius_xy: float) -> Dict[str, float]:
    xy = d.xyz[:, 0:2]
    cxy = center[0:2].reshape(1, 2)
    dist = np.linalg.norm(xy - cxy, axis=1)
    mask = dist <= float(radius_xy)
    if not np.any(mask):
        return {}
    um = d.umag[mask]
    uv = d.uv[mask]
    idx_local = int(np.argmax(um))
    hotspot_uv = uv[idx_local]
    return {
        "n": float(mask.sum()),
        "max": float(np.max(um)),
        "p99": float(np.percentile(um, 99)),
        "mean": float(np.mean(um)),
        "hotspot_u": float(hotspot_uv[0]),
        "hotspot_v": float(hotspot_uv[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--radius", type=float, default=12.0, help="XY radius on mirror (same units as INP, default: 12)")
    ap.add_argument("files", nargs="+", help="Mirror deflection TXT files (results/deflection_*.txt)")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    centers = _load_bolt_centers(cfg)
    if not centers:
        raise SystemExit("No bolt centers resolved.")

    print(f"== Bolt region metrics == radius_xy={float(args.radius):g}")
    for c in centers:
        cc = c.center_xyz
        print(f"- {c.name}: center_xy=({cc[0]:.3f},{cc[1]:.3f}) z={cc[2]:.3f}")

    for f in args.files:
        path = Path(f)
        d = _load_deflection_txt(path)
        print(f"\n== {path.name} ==")
        for c in centers:
            stats = _region_stats(d, c.center_xyz, float(args.radius))
            if not stats:
                print(f"{c.name}: no nodes within radius")
                continue
            print(
                f"{c.name}: n={int(stats['n'])} max|u|={stats['max']:.6e} p99={stats['p99']:.6e} "
                f"mean={stats['mean']:.6e} hotspot_uv=({stats['hotspot_u']:.3f},{stats['hotspot_v']:.3f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

