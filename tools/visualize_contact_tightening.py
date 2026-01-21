#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize contact surfaces and nut tightening rotation from CDB/INP.

Outputs:
  - a static contact surface snapshot (PNG)
  - an animated tightening GIF (optional)
  - a JSON metadata file with axes/centers (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inp_io.cdb_parser import load_cdb
from inp_io.inp_parser import load_inp
from mesh.surface_utils import (
    resolve_surface_to_tris,
    triangulate_part_boundary,
    sample_points_on_surface,
)


def _normalize_axis(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size != 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = v / n
    if np.dot(v, np.array([0.0, 0.0, 1.0])) < 0.0:
        v = -v
    return v


def _auto_axis_from_nodes(nodes_xyz: Dict[int, Tuple[float, float, float]]) -> np.ndarray:
    coords = np.asarray(list(nodes_xyz.values()), dtype=np.float64)
    if coords.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = coords.mean(axis=0)
    Xc = coords - c
    cov = (Xc.T @ Xc) / max(coords.shape[0], 1)
    w, v = np.linalg.eigh(cov)
    axis = v[:, int(np.argmin(w))]
    return _normalize_axis(axis)


def _rotate_points(X: np.ndarray, axis: np.ndarray, center: np.ndarray, theta: float) -> np.ndarray:
    a = _normalize_axis(axis).reshape(1, 3)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    r = X - c
    proj = np.sum(r * a, axis=1, keepdims=True) * a
    radial = r - proj
    ct = np.cos(theta)
    st = np.sin(theta)
    cross = np.cross(a, radial)
    radial_rot = radial * ct + cross * st + a * (np.sum(a * radial, axis=1, keepdims=True)) * (1.0 - ct)
    return c + proj + radial_rot


def _lighten(color: Tuple[float, float, float], amount: float = 0.5) -> Tuple[float, float, float]:
    c = np.asarray(color, dtype=np.float64)
    return tuple(1.0 - (1.0 - c) * amount)


def _sanitize_name(name: str, max_len: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    if not s:
        s = "unnamed"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _tri_surface_to_mesh(
    part,
    tri_node_ids: np.ndarray,
    max_tris: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    tri_ids = np.asarray(tri_node_ids, dtype=np.int64)
    if tri_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    if max_tris is not None:
        max_tris = int(max_tris)
        if max_tris <= 0:
            max_tris = None
    if max_tris is not None and tri_ids.shape[0] > max_tris:
        idx = rng.choice(tri_ids.shape[0], size=max_tris, replace=False)
        tri_ids = tri_ids[idx]
    uniq, inv = np.unique(tri_ids.reshape(-1), return_inverse=True)
    verts = np.array([part.nodes_xyz[int(nid)] for nid in uniq], dtype=np.float64)
    faces = inv.reshape(-1, 3).astype(np.int32)
    return verts, faces


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    if pts.size == 0:
        return
    minv = pts.min(axis=0)
    maxv = pts.max(axis=0)
    center = 0.5 * (minv + maxv)
    radius = 0.5 * float(np.max(maxv - minv))
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _bounds_from_meshes(meshes: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    minv = None
    maxv = None
    for mesh in meshes:
        verts = mesh.get("verts")
        if verts is None or verts.size == 0:
            continue
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        if minv is None:
            minv = vmin.copy()
            maxv = vmax.copy()
        else:
            minv = np.minimum(minv, vmin)
            maxv = np.maximum(maxv, vmax)
    return minv, maxv


def _set_axes_equal_bounds(ax, minv: Optional[np.ndarray], maxv: Optional[np.ndarray]) -> None:
    if minv is None or maxv is None:
        return
    center = 0.5 * (minv + maxv)
    radius = 0.5 * float(np.max(maxv - minv))
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _expand_bounds(
    minv: Optional[np.ndarray],
    maxv: Optional[np.ndarray],
    pad: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if minv is None or maxv is None:
        return minv, maxv
    pad = float(pad)
    if pad <= 1.0:
        return minv, maxv
    center = 0.5 * (minv + maxv)
    half = 0.5 * (maxv - minv) * pad
    return center - half, center + half


def _build_axes(elev: float, azim: float) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    return fig, ax


def _add_mesh(
    ax: plt.Axes,
    verts: np.ndarray,
    faces: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float,
    edgecolor: str = "none",
    linewidths: float = 0.0,
) -> Optional[Poly3DCollection]:
    if verts.size == 0 or faces.size == 0:
        return None
    poly = Poly3DCollection(
        verts[faces],
        facecolor=color,
        edgecolor=edgecolor,
        linewidths=linewidths,
        alpha=float(alpha),
    )
    ax.add_collection3d(poly)
    return poly


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_mesh_path(cfg: Dict[str, Any], config_path: str) -> str:
    mesh_path = (cfg.get("inp_path") or cfg.get("cdb_path") or cfg.get("mesh_path") or "").strip()
    if not mesh_path:
        raise ValueError("config.yaml must provide inp_path/cdb_path/mesh_path.")
    if os.path.isabs(mesh_path) and os.path.exists(mesh_path):
        return mesh_path
    candidates = []
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    if mesh_path:
        candidates.append(os.path.join(cfg_dir, mesh_path))
        candidates.append(os.path.join(ROOT, mesh_path))
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mesh file not found: {mesh_path}")


def _load_asm(mesh_path: str):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".cdb":
        return load_cdb(mesh_path)
    return load_inp(mesh_path)


def _nut_specs_from_cfg(cfg: Dict[str, Any], asm) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for entry in cfg.get("nuts", []) or []:
        specs.append(
            {
                "name": entry.get("name", ""),
                "part": entry.get("part", entry.get("part_name", "")),
                "axis": entry.get("axis", None),
                "center": entry.get("center", None),
            }
        )
    if not specs:
        for pname in asm.parts.keys():
            if "LUOMU" in pname.upper():
                specs.append({"name": pname, "part": pname, "axis": None, "center": None})
    return specs


def _tighten_angles_from_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, str, bool]:
    a_min = float(cfg.get("tighten_angle_min", 0.0))
    a_max = float(cfg.get("tighten_angle_max", 30.0))
    tcfg = cfg.get("tightening_config", {}) or {}
    unit = str(tcfg.get("angle_unit", "deg") or "deg").lower()
    clockwise = bool(tcfg.get("clockwise", True))
    return a_min, a_max, unit, clockwise


def _order_from_cfg(cfg: Dict[str, Any], n_nuts: int) -> List[int]:
    order = list(range(n_nuts))
    seq = cfg.get("preload_sequence", []) or []
    if seq and isinstance(seq[0], dict) and seq[0].get("order") is not None:
        raw = list(seq[0].get("order") or [])
        if raw:
            raw_arr = np.array(raw, dtype=np.int64).reshape(-1)
            if raw_arr.min() >= 1 and raw_arr.max() <= n_nuts:
                raw_arr = raw_arr - 1
            if len(raw_arr) == n_nuts and sorted(raw_arr.tolist()) == list(range(n_nuts)):
                order = raw_arr.tolist()
    return order


def _build_frame_angles(
    n_nuts: int,
    order: Sequence[int],
    a_min: float,
    a_max: float,
    frames_per_nut: int,
) -> List[np.ndarray]:
    angles = np.full((n_nuts,), a_min, dtype=np.float64)
    frames: List[np.ndarray] = []
    for idx in order:
        for t in np.linspace(a_min, a_max, frames_per_nut, dtype=np.float64):
            frame = angles.copy()
            frame[int(idx)] = t
            frames.append(frame)
        angles[int(idx)] = a_max
    frames.append(angles.copy())
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize contact surfaces and nut tightening.")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "results", "contact_surfaces.png"))
    ap.add_argument("--gif", default=os.path.join(ROOT, "results", "contact_tightening_demo.gif"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "results", "contact_tightening_meta.json"))
    ap.add_argument("--contact-dir", default=os.path.join(ROOT, "results", "contact_pairs"))
    ap.add_argument("--max-pairs", type=int, default=0, help="Limit contact pairs (0=all).")
    ap.add_argument("--max-tris", type=int, default=0, help="Max triangles per surface (0=all).")
    ap.add_argument("--solid-max-tris", type=int, default=0, help="Max triangles per solid part (0=all).")
    ap.add_argument("--frames-per-nut", type=int, default=20)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--contact-alpha", type=float, default=0.9)
    ap.add_argument("--contact-edgecolor", default="k")
    ap.add_argument("--contact-edgewidth", type=float, default=0.3)
    ap.add_argument("--solid-alpha", type=float, default=0.25)
    ap.add_argument("--bolt-alpha", type=float, default=0.75)
    ap.add_argument("--nut-alpha", type=float, default=0.98)
    ap.add_argument("--slave-lighten", type=float, default=0.85)
    ap.add_argument("--contact-context", default="pair", choices=["pair", "all"])
    ap.add_argument("--contact-pad", type=float, default=1.2)
    ap.add_argument("--no-contact-zoom", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-contact-images", action="store_true")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    mesh_path = _resolve_mesh_path(cfg, args.config)
    asm = _load_asm(mesh_path)

    rng = np.random.default_rng(args.seed)

    nut_specs = _nut_specs_from_cfg(cfg, asm)
    nut_parts = {spec.get("part") for spec in nut_specs if spec.get("part")}
    bolt_parts = {name for name in asm.parts.keys() if "LUOSHUAN" in name.upper()}

    # Collect solid meshes for each part (entity view)
    part_mesh_by_name: Dict[str, Dict[str, Any]] = {}
    all_part_meshes: List[Dict[str, Any]] = []
    for part_name, part in asm.parts.items():
        if part_name == "__CONTACT__":
            continue
        try:
            ts = triangulate_part_boundary(part, part_name, log_summary=False)
            verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.solid_max_tris, rng)
            if verts.size == 0 or faces.size == 0:
                continue
            mesh = {"name": part_name, "verts": verts, "faces": faces}
            part_mesh_by_name[part_name] = mesh
            all_part_meshes.append(mesh)
        except Exception as exc:
            print(f"[warn] skip solid part {part_name}: {exc}")

    # Collect contact surfaces
    pairs = list(getattr(asm, "contact_pairs", []) or [])
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    contact_mesh_by_name: Dict[str, Dict[str, Any]] = {}
    contact_surfs = []
    cmap = plt.get_cmap("tab20")
    for i, pair in enumerate(pairs):
        base_color = cmap(i % cmap.N)[:3]
        for role, key in (("master", pair.master), ("slave", pair.slave)):
            mesh = contact_mesh_by_name.get(key)
            if mesh is None:
                try:
                    ts = resolve_surface_to_tris(asm, key, log_summary=False)
                    part = asm.parts.get(ts.part_name)
                    if part is None:
                        continue
                    verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.max_tris, rng)
                    if verts.size == 0 or faces.size == 0:
                        continue
                    mesh = {"name": key, "part": ts.part_name, "verts": verts, "faces": faces}
                    contact_mesh_by_name[key] = mesh
                except Exception as exc:
                    print(f"[warn] skip contact surface {key}: {exc}")
                    continue
            if mesh is None:
                continue
            color = base_color if role == "master" else _lighten(base_color, args.slave_lighten)
            contact_surfs.append(
                {"name": key, "role": role, "verts": mesh["verts"], "faces": mesh["faces"], "color": color}
            )

    bounds_meshes = list(all_part_meshes) + list(contact_mesh_by_name.values())
    bounds_min, bounds_max = _bounds_from_meshes(bounds_meshes)

    # Per-contact images (one contact pair per image)
    if pairs and not args.no_contact_images:
        out_dir = args.contact_dir
        os.makedirs(out_dir, exist_ok=True)
        manifest = []
        solid_color = (0.78, 0.78, 0.78)
        for i, pair in enumerate(pairs):
            master_raw = pair.master or ""
            slave_raw = pair.slave or ""
            master_name = _sanitize_name(master_raw)
            slave_name = _sanitize_name(slave_raw)
            master_part = None
            slave_part = None
            master_mesh = contact_mesh_by_name.get(master_raw)
            slave_mesh = contact_mesh_by_name.get(slave_raw)
            if master_mesh is not None:
                master_part = master_mesh.get("part")
            if slave_mesh is not None:
                slave_part = slave_mesh.get("part")
            fig, ax = _build_axes(args.elev, args.azim)
            if args.contact_context == "all":
                context_meshes = all_part_meshes
            else:
                context_parts = {p for p in (master_part, slave_part) if p}
                context_meshes = [
                    part_mesh_by_name[pname]
                    for pname in sorted(context_parts)
                    if pname in part_mesh_by_name
                ]
                if not context_meshes:
                    context_meshes = all_part_meshes
            for mesh in context_meshes:
                _add_mesh(
                    ax,
                    mesh["verts"],
                    mesh["faces"],
                    solid_color,
                    args.solid_alpha,
                    edgecolor="none",
                )
            base_color = cmap(i % cmap.N)[:3]
            for role, key in (("master", pair.master), ("slave", pair.slave)):
                mesh = contact_mesh_by_name.get(key)
                if mesh is None:
                    continue
                color = base_color if role == "master" else _lighten(base_color, args.slave_lighten)
                _add_mesh(
                    ax,
                    mesh["verts"],
                    mesh["faces"],
                    color,
                    args.contact_alpha,
                    edgecolor=args.contact_edgecolor,
                    linewidths=args.contact_edgewidth,
                )
            if args.no_contact_zoom:
                _set_axes_equal_bounds(ax, bounds_min, bounds_max)
            else:
                pair_meshes = [
                    m
                    for m in (master_mesh, slave_mesh)
                    if m is not None and m.get("verts") is not None
                ]
                pair_min, pair_max = _bounds_from_meshes(pair_meshes)
                if pair_min is None:
                    pair_min, pair_max = _bounds_from_meshes(context_meshes)
                pair_min, pair_max = _expand_bounds(pair_min, pair_max, args.contact_pad)
                _set_axes_equal_bounds(ax, pair_min, pair_max)
            out_path = os.path.join(
                out_dir,
                f"contact_{i + 1:03d}_{master_name}__{slave_name}.png",
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            manifest.append(
                {
                    "index": i + 1,
                    "master": master_raw,
                    "slave": slave_raw,
                    "image": os.path.basename(out_path),
                }
            )
        manifest_path = os.path.join(out_dir, "contact_pairs.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[ok] contact images saved: {out_dir}")
        print(f"[ok] contact manifest saved: {manifest_path}")

    # Collect nuts
    nut_meshes = []
    meta = {"nuts": []}
    if not nut_specs:
        print("[warn] No nut parts found (LUOMU*).")

    n_points_each = int(cfg.get("tightening_n_points_each", cfg.get("preload_n_points_each", 800)))
    for i, spec in enumerate(nut_specs):
        part_name = spec.get("part") or ""
        if part_name not in asm.parts:
            print(f"[warn] nut part not found: {part_name}")
            continue
        part = asm.parts[part_name]
        ts = triangulate_part_boundary(part, part_name, log_summary=False)
        verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.max_tris, rng)
        if verts.size == 0 or faces.size == 0:
            fallback = part_mesh_by_name.get(part_name)
            if fallback is None:
                print(f"[warn] nut surface empty: {part_name}")
                continue
            verts, faces = fallback["verts"], fallback["faces"]

        axis = spec.get("axis")
        if axis is None:
            axis = _auto_axis_from_nodes(part.nodes_xyz)
        else:
            axis = _normalize_axis(axis)

        center = spec.get("center")
        if center is None:
            if ts.tri_node_ids.size > 0:
                X, _, _, _ = sample_points_on_surface(part, ts, n_points_each, rng=rng)
                center = X.mean(axis=0)
            else:
                coords = np.asarray(list(part.nodes_xyz.values()), dtype=np.float64)
                center = coords.mean(axis=0) if coords.size else np.zeros((3,), dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)

        bbox = verts.max(axis=0) - verts.min(axis=0)
        axis_len = float(max(bbox.max(), 1.0)) * 0.6

        nut_meshes.append(
            {
                "name": spec.get("name") or part_name,
                "part": part_name,
                "verts": verts,
                "faces": faces,
                "axis": axis,
                "center": center,
                "axis_len": axis_len,
            }
        )
        meta["nuts"].append(
            {
                "name": spec.get("name") or part_name,
                "part": part_name,
                "axis": axis.tolist(),
                "center": center.tolist(),
            }
        )

    if nut_meshes:
        nut_min, nut_max = _bounds_from_meshes(nut_meshes)
        if nut_min is not None:
            if bounds_min is None:
                bounds_min, bounds_max = nut_min, nut_max
            else:
                bounds_min = np.minimum(bounds_min, nut_min)
                bounds_max = np.maximum(bounds_max, nut_max)

    if args.meta:
        os.makedirs(os.path.dirname(args.meta), exist_ok=True)
        with open(args.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Build figure
    base_meshes = [
        mesh
        for name, mesh in part_mesh_by_name.items()
        if name not in nut_parts and name not in bolt_parts
    ]
    bolt_meshes = [part_mesh_by_name[name] for name in sorted(bolt_parts) if name in part_mesh_by_name]

    fig, ax = _build_axes(args.elev, args.azim)
    solid_color = (0.78, 0.78, 0.78)
    bolt_color = (0.45, 0.45, 0.45)

    all_pts = []
    for mesh in base_meshes:
        _add_mesh(
            ax,
            mesh["verts"],
            mesh["faces"],
            solid_color,
            args.solid_alpha,
            edgecolor="none",
        )
        all_pts.append(mesh["verts"])
    for mesh in bolt_meshes:
        _add_mesh(
            ax,
            mesh["verts"],
            mesh["faces"],
            bolt_color,
            args.bolt_alpha,
            edgecolor="none",
        )
        all_pts.append(mesh["verts"])
    for surf in contact_surfs:
        _add_mesh(
            ax,
            surf["verts"],
            surf["faces"],
            surf["color"],
            args.contact_alpha,
            edgecolor=args.contact_edgecolor,
            linewidths=args.contact_edgewidth,
        )
        all_pts.append(surf["verts"])

    nut_polys = []
    nut_cmap = plt.get_cmap("tab10")
    for i, nut in enumerate(nut_meshes):
        color = nut_cmap(i % nut_cmap.N)[:3]
        poly = _add_mesh(
            ax,
            nut["verts"],
            nut["faces"],
            color,
            args.nut_alpha,
            edgecolor="k",
            linewidths=0.1,
        )
        if poly is not None:
            nut_polys.append(poly)
            nut["poly"] = poly
        all_pts.append(nut["verts"])

        # Axis line
        axis = nut["axis"]
        center = nut["center"]
        axis_len = nut["axis_len"]
        p0 = center - axis * axis_len
        p1 = center + axis * axis_len
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=2.0)

    if bounds_min is not None:
        _set_axes_equal_bounds(ax, bounds_min, bounds_max)
    elif all_pts:
        _set_axes_equal(ax, np.vstack(all_pts))

    text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes)

    # Snapshot
    if args.snapshot:
        os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
        fig.savefig(args.snapshot, dpi=args.dpi, bbox_inches="tight")
        print(f"[ok] snapshot saved: {args.snapshot}")

    # Animation
    if args.no_gif or not args.gif:
        return

    a_min, a_max, unit, clockwise = _tighten_angles_from_cfg(cfg)
    order = _order_from_cfg(cfg, len(nut_meshes))
    frames = _build_frame_angles(len(nut_meshes), order, a_min, a_max, args.frames_per_nut)

    def _angle_to_rad(theta: float) -> float:
        if unit.startswith("deg"):
            return np.deg2rad(theta)
        return float(theta)

    def _update(frame_idx: int):
        angles = frames[frame_idx]
        for nut, theta in zip(nut_meshes, angles):
            t = _angle_to_rad(float(theta))
            if clockwise:
                t = -t
            v_rot = _rotate_points(nut["verts"], nut["axis"], nut["center"], t)
            poly = nut.get("poly")
            if poly is not None:
                poly.set_verts(v_rot[nut["faces"]])
        angle_txt = ", ".join(f"{float(a):.2f}" for a in angles.tolist())
        text.set_text(f"theta=[{angle_txt}] {unit}")
        return nut_polys + [text]

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / args.fps, blit=False)
        os.makedirs(os.path.dirname(args.gif), exist_ok=True)
        anim.save(args.gif, writer=PillowWriter(fps=args.fps))
        print(f"[ok] animation saved: {args.gif}")
    except Exception as exc:
        print(f"[warn] GIF save failed ({exc}); falling back to PNG frames.")
        out_dir = os.path.splitext(args.gif)[0] + "_frames"
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(frames)):
            _update(i)
            fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"), dpi=args.dpi, bbox_inches="tight")
        print(f"[ok] frames saved: {out_dir}")


if __name__ == "__main__":
    main()
