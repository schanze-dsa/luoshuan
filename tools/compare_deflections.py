#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare mirror deflection exports (results/deflection_*.txt).

These TXT files are written by `src/viz/mirror_viz.py` and contain per-node
displacements on the mirror surface:
  node_id x y z u_x u_y u_z |u| u_plane v_plane

This tool helps answer "these cloud maps look identical" by printing numeric
differences (and optionally saving simple scatter diff plots).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class DeflectionData:
    path: Path
    node_id: np.ndarray  # (N,) int64
    xyz: np.ndarray  # (N,3) float64
    u: np.ndarray  # (N,3) float64
    umag: np.ndarray  # (N,) float64
    uv: np.ndarray  # (N,2) float64


def _load_txt(path: Path) -> DeflectionData:
    node_id = []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            node_id.append(int(parts[0]))
            rows.append([float(x) for x in parts[1:10]])

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")

    node_id_arr = np.asarray(node_id, dtype=np.int64)
    data = np.asarray(rows, dtype=np.float64)
    order = np.argsort(node_id_arr)
    node_id_arr = node_id_arr[order]
    data = data[order]

    xyz = data[:, 0:3]
    u = data[:, 3:6]
    umag = data[:, 6]
    uv = data[:, 7:9]

    return DeflectionData(path=path, node_id=node_id_arr, xyz=xyz, u=u, umag=umag, uv=uv)


def _ensure_same_nodes(a: DeflectionData, b: DeflectionData) -> None:
    if a.node_id.shape != b.node_id.shape or not np.array_equal(a.node_id, b.node_id):
        raise ValueError(
            "Node IDs do not match between files:\n"
            f"  A={a.path}\n"
            f"  B={b.path}\n"
            "Tip: ensure both TXT files were exported from the same surface mesh."
        )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-30)
    return float((x * y).sum() / denom)


def _summarize_one(d: DeflectionData) -> str:
    idx = int(np.argmax(d.umag))
    nid = int(d.node_id[idx])
    u, v = d.uv[idx]
    thr = 0.95 * float(d.umag[idx])
    mask = d.umag >= thr
    if np.any(mask):
        uv_c = d.uv[mask].mean(axis=0)
        cen_txt = f" hotspot_uv≈({uv_c[0]:.3f},{uv_c[1]:.3f})"
    else:
        cen_txt = ""
    return (
        f"{d.path.name}: max|u|={d.umag[idx]:.6e} at node={nid} (u,v)=({u:.3f},{v:.3f})"
        f"{cen_txt}"
    )


def _pair_metrics(a: DeflectionData, b: DeflectionData) -> str:
    _ensure_same_nodes(a, b)
    du = b.u - a.u
    du_mag = np.linalg.norm(du, axis=1)
    dd = b.umag - a.umag

    corr = _pearson(a.umag, b.umag)
    eps = 1e-12
    rel = np.median(du_mag / (np.maximum(a.umag, b.umag) + eps))

    a_max = int(np.argmax(a.umag))
    b_max = int(np.argmax(b.umag))
    uv_dist = float(np.linalg.norm(a.uv[a_max] - b.uv[b_max]))

    def _hotspot_uv(d: DeflectionData, ratio: float = 0.95) -> Optional[np.ndarray]:
        if d.umag.size == 0:
            return None
        m = float(np.max(d.umag))
        if not np.isfinite(m) or m <= 0:
            return None
        mask = d.umag >= ratio * m
        if not np.any(mask):
            return None
        return d.uv[mask].mean(axis=0)

    ha = _hotspot_uv(a)
    hb = _hotspot_uv(b)
    hotspot_dist = None
    if ha is not None and hb is not None:
        hotspot_dist = float(np.linalg.norm(ha - hb))

    return (
        f"{a.path.name} vs {b.path.name}: "
        f"max|du|={du_mag.max():.6e}, mean|du|={du_mag.mean():.6e}, "
        f"max|d|u||={np.abs(dd).max():.6e}, corr(|u|)={corr:.4f}, "
        f"median rel |du|={rel:.4f}, "
        f"argmax_uv_dist={uv_dist:.3f}"
        + ("" if hotspot_dist is None else f", hotspot_uv_dist≈{hotspot_dist:.3f}")
    )


def _save_scatter(path: Path, uv: np.ndarray, values: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=values, s=2.0, cmap="turbo", linewidths=0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u (best-fit plane)")
    ax.set_ylabel("v (best-fit plane)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _iter_txt_paths(items: Iterable[str]) -> Tuple[Path, ...]:
    out = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.txt")))
        else:
            out.append(p)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="TXT files (or directories) to compare")
    ap.add_argument(
        "--baseline",
        default=None,
        help="Baseline TXT; default: first file after expansion",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Write scatter diff plots to --out-dir",
    )
    ap.add_argument(
        "--out-dir",
        default="results/compare",
        help="Output directory for plots (default: results/compare)",
    )
    args = ap.parse_args()

    paths = _iter_txt_paths(args.files)
    if not paths:
        raise SystemExit("No TXT files found.")

    baseline_path = Path(args.baseline) if args.baseline else paths[0]
    baseline = _load_txt(baseline_path)

    others = [p for p in paths if Path(p) != baseline_path]
    if not others:
        raise SystemExit("Need at least 2 files to compare.")

    print(_summarize_one(baseline))
    for p in others:
        d = _load_txt(Path(p))
        print(_summarize_one(d))
        print(_pair_metrics(baseline, d))

        if args.plots:
            _ensure_same_nodes(baseline, d)
            du_mag = np.linalg.norm(d.u - baseline.u, axis=1)
            out_path = Path(args.out_dir) / f"diff__{baseline.path.stem}__{d.path.stem}.png"
            _save_scatter(out_path, baseline.uv, du_mag, f"|du|  ({baseline.path.name} -> {d.path.name})")
            print(f"[plot] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
