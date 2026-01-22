#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Plotly HTML viewer for solids, contact pairs, and nut tightening.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import yaml

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception as exc:
    print(f"[error] Plotly not available: {exc}")
    print("Install with: pip install plotly")
    raise SystemExit(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inp_io.cdb_parser import load_cdb
from inp_io.inp_parser import load_inp
from mesh.surface_utils import resolve_surface_to_tris, triangulate_part_boundary, sample_points_on_surface


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_mesh_path(cfg: Dict[str, Any], config_path: str) -> str:
    mesh_path = (cfg.get("inp_path") or cfg.get("cdb_path") or cfg.get("mesh_path") or "").strip()
    if not mesh_path:
        raise ValueError("config.yaml must provide inp_path/cdb_path/mesh_path.")
    if os.path.isabs(mesh_path) and os.path.exists(mesh_path):
        return mesh_path
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = [os.path.join(cfg_dir, mesh_path), os.path.join(ROOT, mesh_path)]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mesh file not found: {mesh_path}")


def _load_asm(mesh_path: str):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".cdb":
        return load_cdb(mesh_path)
    return load_inp(mesh_path)


def _normalize_contact_pairs(obj: Any) -> List[Tuple[str, str]]:
    if obj is None:
        return []
    seq = obj
    if isinstance(obj, dict):
        seq = [obj]
    elif not isinstance(obj, (list, tuple)):
        seq = [obj]
    out: List[Tuple[str, str]] = []
    for item in seq:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
            continue
        if isinstance(item, dict):
            keys = {str(k).lower(): v for k, v in item.items()}
            m = keys.get("master_key") or keys.get("master") or keys.get("a")
            s = keys.get("slave_key") or keys.get("slave") or keys.get("b")
            if m and s:
                out.append((str(m), str(s)))
            continue
        m = getattr(item, "master", None) or getattr(item, "master_key", None)
        s = getattr(item, "slave", None) or getattr(item, "slave_key", None)
        if m and s:
            out.append((str(m), str(s)))
    return out


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


def _tri_surface_to_mesh(
    part,
    tri_node_ids: np.ndarray,
    max_tris: int,
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


def _lighten(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    out = []
    for c in color:
        out.append(int(round(255 - (255 - int(c)) * amount)))
    return tuple(out)


def _rgb(color: Tuple[int, int, int]) -> str:
    r, g, b = color
    return f"rgb({int(r)},{int(g)},{int(b)})"


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


def _color_cycle() -> List[Tuple[int, int, int]]:
    return [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]


def _make_part_color_map(names: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    unique = sorted({n for n in names if n})
    if not unique:
        return {}
    n = len(unique)
    out: Dict[str, Tuple[int, int, int]] = {}
    for i, name in enumerate(unique):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        out[name] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def _build_vertex_neighbors(n_verts: int, faces: np.ndarray) -> List[np.ndarray]:
    neighbors: List[set] = [set() for _ in range(int(n_verts))]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return [np.fromiter(nbrs, dtype=np.int64) for nbrs in neighbors]


def _laplacian_step(verts: np.ndarray, neighbors: List[np.ndarray], lam: float) -> np.ndarray:
    out = verts.copy()
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        mean = verts[nbrs].mean(axis=0)
        out[i] = verts[i] + lam * (mean - verts[i])
    return out


def _taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iters: int,
    lam: float,
    mu: float,
) -> np.ndarray:
    if iters <= 0:
        return verts
    neighbors = _build_vertex_neighbors(verts.shape[0], faces)
    v = verts.copy()
    for _ in range(int(iters)):
        v = _laplacian_step(v, neighbors, lam)
        v = _laplacian_step(v, neighbors, mu)
    return v


def _bounds_from_verts(verts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if verts.size == 0:
        return None
    return verts.min(axis=0), verts.max(axis=0)


def _merge_bounds(bounds: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for b in bounds:
        if b is None:
            continue
        mins.append(b[0])
        maxs.append(b[1])
    if not mins:
        return None
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def _pad_bounds(bounds: Tuple[np.ndarray, np.ndarray], frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    min_v, max_v = bounds
    span = max_v - min_v
    pad = np.maximum(span * float(frac), 1e-9)
    return min_v - pad, max_v + pad


def _hover_template(
    title: str,
    group: str,
    part: str,
    n_nodes: int,
    n_tris: int,
    material: Optional[str] = None,
    role: Optional[str] = None,
    surface: Optional[str] = None,
) -> str:
    lines = [
        f"<b>{title}</b>",
        f"Group: {group}",
        f"Part: {part}",
        f"Nodes: {n_nodes}",
        f"Tris: {n_tris}",
    ]
    if material:
        lines.append(f"Material: {material}")
    if role:
        lines.append(f"Role: {role}")
    if surface:
        lines.append(f"Surface: {surface}")
    return "<br>".join(lines) + "<extra></extra>"


def _clip_faces_to_cylindrical_bounds(
    verts: np.ndarray,
    faces: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if verts.size == 0 or faces.size == 0:
        return verts, faces
    face_xyz = verts[faces]
    centroids = face_xyz.mean(axis=1)
    r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    keep = (
        (r >= r_min - margin)
        & (r <= r_max + margin)
        & (centroids[:, 2] >= z_min - margin)
        & (centroids[:, 2] <= z_max + margin)
    )
    if not np.any(keep):
        return verts, faces
    faces_kept = faces[keep]
    uniq, inv = np.unique(faces_kept.reshape(-1), return_inverse=True)
    return verts[uniq], inv.reshape(-1, 3)


def _mesh_trace(
    name: str,
    verts: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int],
    opacity: float,
    showlegend: bool = False,
    legendgroup: Optional[str] = None,
    legendgrouptitle: Optional[str] = None,
    hovertemplate: Optional[str] = None,
) -> Optional[go.Mesh3d]:
    if verts.size == 0 or faces.size == 0:
        return None
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=_rgb(color),
        opacity=float(opacity),
        flatshading=False,
        lighting={"ambient": 0.4, "diffuse": 0.8, "specular": 0.3, "roughness": 0.4, "fresnel": 0.2},
        showscale=False,
        showlegend=showlegend,
        legendgroup=legendgroup,
        legendgrouptitle={"text": legendgrouptitle} if legendgrouptitle else None,
        hovertemplate=hovertemplate,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plotly HTML viewer for contact pairs and tightening.")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--out-html", default=os.path.join(ROOT, "results", "contact_viewer.html"))
    ap.add_argument("--out-meta", default=os.path.join(ROOT, "results", "contact_viewer_meta.json"))
    ap.add_argument("--max-pairs", type=int, default=0, help="Limit contact pairs (0=all).")
    ap.add_argument("--max-tris", type=int, default=0, help="Max triangles per contact surface (0=all).")
    ap.add_argument("--solid-max-tris", type=int, default=0, help="Max triangles per solid part (0=all).")
    ap.add_argument("--frames-per-nut", type=int, default=8)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--contact-alpha", type=float, default=1.0)
    ap.add_argument("--solid-alpha", type=float, default=1.0)
    ap.add_argument("--bolt-alpha", type=float, default=1.0)
    ap.add_argument("--nut-alpha", type=float, default=1.0)
    ap.add_argument("--slave-lighten", type=float, default=0.85)
    ap.add_argument("--smooth-iters", type=int, default=1, help="Taubin smoothing iterations (0=off).")
    ap.add_argument("--smooth-lambda", type=float, default=0.45)
    ap.add_argument("--smooth-mu", type=float, default=-0.47)
    ap.add_argument("--smooth-target", default="all", choices=["all", "solids", "none"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="Contact Viewer")
    ap.add_argument("--inline", action="store_true", help="Inline plotly.js (offline use).")
    ap.add_argument("--cdn", action="store_true", help="Load plotly.js from CDN.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    mesh_path = _resolve_mesh_path(cfg, args.config)
    asm = _load_asm(mesh_path)

    rng = np.random.default_rng(args.seed)

    part2mat_raw = cfg.get("part2mat", {}) or {}
    part2mat = {str(k).upper(): str(v) for k, v in part2mat_raw.items()}

    def _material_for_part(name: str) -> str:
        return part2mat.get(str(name).upper(), "")

    node_to_parts: Dict[int, List[str]] = {}
    for part_name, part in asm.parts.items():
        if part_name == "__CONTACT__":
            continue
        for nid in part.node_ids:
            node_to_parts.setdefault(int(nid), []).append(part_name)

    if args.solid_max_tris == 0 or args.max_tris == 0:
        print("[info] Full geometry enabled (no triangle decimation). HTML may be very large.")

    nut_specs = _nut_specs_from_cfg(cfg, asm)
    nut_parts = {spec.get("part") for spec in nut_specs if spec.get("part")}
    bolt_parts = {name for name in asm.parts.keys() if "LUOSHUAN" in name.upper()}

    # Build solid meshes
    part_mesh_by_name: Dict[str, Dict[str, Any]] = {}
    for part_name, part in asm.parts.items():
        if part_name == "__CONTACT__":
            continue
        ts = triangulate_part_boundary(part, part_name, log_summary=False)
        verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.solid_max_tris, rng)
        if verts.size == 0 or faces.size == 0:
            continue
        if args.smooth_target in ("all", "solids"):
            verts = _taubin_smooth(verts, faces, args.smooth_iters, args.smooth_lambda, args.smooth_mu)
        part_mesh_by_name[part_name] = {
            "name": part_name,
            "verts": verts,
            "faces": faces,
            "n_nodes": int(verts.shape[0]),
            "n_tris": int(faces.shape[0]),
            "bounds": _bounds_from_verts(verts),
        }

    base_meshes = [
        part_mesh_by_name[name]
        for name in sorted(part_mesh_by_name.keys())
        if name not in nut_parts and name not in bolt_parts
    ]
    bolt_meshes = [part_mesh_by_name[name] for name in sorted(bolt_parts) if name in part_mesh_by_name]
    part_colors = _make_part_color_map(part_mesh_by_name.keys())

    # Build nut meshes + axes
    nut_meshes: List[Dict[str, Any]] = []
    n_points_each = int(cfg.get("tightening_n_points_each", cfg.get("preload_n_points_each", 800)))
    for spec in nut_specs:
        part_name = spec.get("part") or ""
        if part_name not in asm.parts:
            continue
        part = asm.parts[part_name]
        base_mesh = part_mesh_by_name.get(part_name)
        if base_mesh is None:
            continue
        axis = spec.get("axis")
        if axis is None:
            axis = _auto_axis_from_nodes(part.nodes_xyz)
        else:
            axis = _normalize_axis(axis)
        center = spec.get("center")
        if center is None:
            ts = triangulate_part_boundary(part, part_name, log_summary=False)
            if ts.tri_node_ids.size > 0:
                X, _, _, _ = sample_points_on_surface(part, ts, n_points_each, rng=rng)
                center = X.mean(axis=0)
            else:
                coords = np.asarray(list(part.nodes_xyz.values()), dtype=np.float64)
                center = coords.mean(axis=0) if coords.size else np.zeros((3,), dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        nut_meshes.append(
            {
                "name": spec.get("name") or part_name,
                "part": part_name,
                "verts": base_mesh["verts"],
                "faces": base_mesh["faces"],
                "axis": axis,
                "center": center,
            }
        )

    # Contact meshes
    pairs = list(getattr(asm, "contact_pairs", []) or [])
    cfg_pairs = _normalize_contact_pairs(cfg.get("contact_pairs"))
    if cfg_pairs:
        allow = {(m, s) for m, s in cfg_pairs}
        pairs = [pair for pair in pairs if (pair.master, pair.slave) in allow]
        if not pairs:
            print("[warn] Config contact_pairs did not match any CDB pairs.")
            available = [f"{p.master} -> {p.slave}" for p in getattr(asm, "contact_pairs", []) or []]
            if available:
                print("[warn] Available pairs:")
                for entry in available:
                    print(f"  - {entry}")
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    contact_mesh_by_name: Dict[str, Dict[str, Any]] = {}
    for pair in pairs:
        for key in (pair.master, pair.slave):
            if key in contact_mesh_by_name:
                continue
            ts = resolve_surface_to_tris(asm, key, log_summary=False)
            part = asm.parts.get(ts.part_name)
            if part is None:
                continue
            uniq_nodes = np.unique(ts.tri_node_ids.reshape(-1))
            counts = Counter()
            for nid in uniq_nodes.tolist():
                for owner in node_to_parts.get(int(nid), []):
                    counts[owner] += 1
            owner_part = counts.most_common(1)[0][0] if counts else ts.part_name
            verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.max_tris, rng)
            if verts.size == 0 or faces.size == 0:
                continue
            if args.smooth_target == "all":
                verts = _taubin_smooth(verts, faces, args.smooth_iters, args.smooth_lambda, args.smooth_mu)
            contact_mesh_by_name[key] = {
                "name": key,
                "part": owner_part,
                "verts": verts,
                "faces": faces,
                "n_nodes": int(verts.shape[0]),
                "n_tris": int(faces.shape[0]),
                "bounds": _bounds_from_verts(verts),
                "r_min": float(np.sqrt((verts[:, 0] ** 2 + verts[:, 1] ** 2)).min()),
                "r_max": float(np.sqrt((verts[:, 0] ** 2 + verts[:, 1] ** 2)).max()),
                "z_min": float(verts[:, 2].min()),
                "z_max": float(verts[:, 2].max()),
            }

    global_bounds = _merge_bounds(
        [m.get("bounds") for m in part_mesh_by_name.values()]
        + [m.get("bounds") for m in contact_mesh_by_name.values()]
    )
    if global_bounds is None:
        global_bounds = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    global_bounds = _pad_bounds(global_bounds, 0.05)

    pair_bounds: List[Tuple[np.ndarray, np.ndarray]] = []
    for pair in pairs:
        bounds = _merge_bounds(
            [
                contact_mesh_by_name.get(pair.master, {}).get("bounds"),
                contact_mesh_by_name.get(pair.slave, {}).get("bounds"),
            ]
        )
        if bounds is None:
            bounds = global_bounds
        pair_bounds.append(_pad_bounds(bounds, 0.08))

    # Build traces
    traces: List[go.Mesh3d] = []
    trace_names: List[str] = []
    base_indices: List[int] = []
    bolt_indices: List[int] = []
    nut_indices: List[int] = []
    contact_indices: List[int] = []
    contact_pair_to_indices: List[List[int]] = []

    color_cycle = _color_cycle()
    legend_group_seen: Set[str] = set()

    def _legend_title(group: str) -> Optional[str]:
        if group in legend_group_seen:
            return None
        legend_group_seen.add(group)
        return group

    for i, mesh in enumerate(base_meshes):
        color = part_colors.get(mesh["name"], color_cycle[i % len(color_cycle)])
        material = _material_for_part(mesh["name"])
        hover = _hover_template(
            mesh["name"],
            "Solids",
            mesh["name"],
            mesh["n_nodes"],
            mesh["n_tris"],
            material=material,
        )
        trace = _mesh_trace(
            mesh["name"],
            mesh["verts"],
            mesh["faces"],
            color,
            args.solid_alpha,
            showlegend=True,
            legendgroup="Solids",
            legendgrouptitle=_legend_title("Solids"),
            hovertemplate=hover,
        )
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(mesh["name"])
        base_indices.append(len(traces) - 1)

    for i, mesh in enumerate(bolt_meshes):
        color = part_colors.get(mesh["name"], color_cycle[i % len(color_cycle)])
        material = _material_for_part(mesh["name"])
        hover = _hover_template(
            mesh["name"],
            "Bolts",
            mesh["name"],
            mesh["n_nodes"],
            mesh["n_tris"],
            material=material,
        )
        trace = _mesh_trace(
            mesh["name"],
            mesh["verts"],
            mesh["faces"],
            color,
            args.bolt_alpha,
            showlegend=True,
            legendgroup="Bolts",
            legendgrouptitle=_legend_title("Bolts"),
            hovertemplate=hover,
        )
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(mesh["name"])
        bolt_indices.append(len(traces) - 1)

    a_min, a_max, unit, clockwise = _tighten_angles_from_cfg(cfg)
    unit_rad = unit.startswith("deg")
    for i, nut in enumerate(nut_meshes):
        color = part_colors.get(nut["part"], color_cycle[i % len(color_cycle)])
        angle = a_min
        if unit_rad:
            angle = np.deg2rad(angle)
        angle = -angle if clockwise else angle
        v0 = _rotate_points(nut["verts"], nut["axis"], nut["center"], float(angle))
        material = _material_for_part(nut["part"])
        hover = _hover_template(
            nut["name"],
            "Nuts",
            nut["part"],
            int(nut["verts"].shape[0]),
            int(nut["faces"].shape[0]),
            material=material,
        )
        trace = _mesh_trace(
            nut["name"],
            v0,
            nut["faces"],
            color,
            args.nut_alpha,
            showlegend=True,
            legendgroup="Nuts",
            legendgrouptitle=_legend_title("Nuts"),
            hovertemplate=hover,
        )
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(nut["name"])
        nut_indices.append(len(traces) - 1)

    for i, pair in enumerate(pairs):
        pair_indices: List[int] = []
        pair_color = color_cycle[i % len(color_cycle)]
        # Clip the larger surface to the smaller surface's radial/z range for visual match.
        r_margin = 0.02
        r_min = r_max = z_min = z_max = None
        if pair.master in contact_mesh_by_name and pair.slave in contact_mesh_by_name:
            m = contact_mesh_by_name[pair.master]
            s = contact_mesh_by_name[pair.slave]
            span_m = max(m["r_max"] - m["r_min"], 1e-9)
            span_s = max(s["r_max"] - s["r_min"], 1e-9)
            if span_m <= span_s:
                r_min, r_max, z_min, z_max = m["r_min"], m["r_max"], m["z_min"], m["z_max"]
            else:
                r_min, r_max, z_min, z_max = s["r_min"], s["r_max"], s["z_min"], s["z_max"]
            r_margin = 0.05 * max(r_max - r_min, 1e-9)
        for role, key in (("master", pair.master), ("slave", pair.slave)):
            mesh = contact_mesh_by_name.get(key)
            if mesh is None:
                continue
            color = pair_color if role == "master" else _lighten(pair_color, args.slave_lighten)
            verts = mesh["verts"]
            faces = mesh["faces"]
            if r_min is not None:
                span = mesh["r_max"] - mesh["r_min"]
                clip_needed = span > (r_max - r_min) * 1.15
                if clip_needed:
                    verts, faces = _clip_faces_to_cylindrical_bounds(
                        verts,
                        faces,
                        r_min,
                        r_max,
                        z_min,
                        z_max,
                        r_margin,
                    )
            material = _material_for_part(mesh["part"])
            hover = _hover_template(
                f"{key} ({role})",
                "Contacts",
                mesh["part"],
                int(verts.shape[0]),
                int(faces.shape[0]),
                material=material,
                role=role,
                surface=key,
            )
            trace = _mesh_trace(
                f"{key} ({role})",
                verts,
                faces,
                color,
                args.contact_alpha,
                showlegend=True,
                legendgroup="Contacts",
                legendgrouptitle=_legend_title("Contacts"),
                hovertemplate=hover,
            )
            if trace is None:
                continue
            traces.append(trace)
            trace_names.append(f"{key}:{role}")
            idx = len(traces) - 1
            contact_indices.append(idx)
            pair_indices.append(idx)
        contact_pair_to_indices.append(pair_indices)

    # Animation frames
    frames: List[go.Frame] = []
    if nut_meshes and nut_indices:
        order = _order_from_cfg(cfg, len(nut_meshes))
        frame_angles = _build_frame_angles(len(nut_meshes), order, a_min, a_max, args.frames_per_nut)

        for fi, angles in enumerate(frame_angles):
            data = []
            for nut, theta in zip(nut_meshes, angles.tolist()):
                t = float(theta)
                if unit_rad:
                    t = np.deg2rad(t)
                if clockwise:
                    t = -t
                v_rot = _rotate_points(nut["verts"], nut["axis"], nut["center"], t)
                data.append(
                    {
                        "type": "mesh3d",
                        "x": v_rot[:, 0],
                        "y": v_rot[:, 1],
                        "z": v_rot[:, 2],
                    }
                )
            frames.append(go.Frame(data=data, name=f"f{fi:04d}", traces=nut_indices))

    # Contact pair visibility buttons
    n_traces = len(traces)
    always_on = set(base_indices + bolt_indices + nut_indices)
    all_contacts_visible = [i in always_on or i in contact_indices for i in range(n_traces)]

    def _scene_range_layout(bounds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        bmin, bmax = bounds
        return {
            "scene": {
                "xaxis": {"range": [float(bmin[0]), float(bmax[0])]},
                "yaxis": {"range": [float(bmin[1]), float(bmax[1])]},
                "zaxis": {"range": [float(bmin[2]), float(bmax[2])]},
            }
        }

    buttons = [
        {
            "label": "All Pairs",
            "method": "update",
            "args": [
                {"visible": all_contacts_visible},
                {"title": args.title + " - All Contact Pairs", **_scene_range_layout(global_bounds)},
            ],
        }
    ]

    for i, pair in enumerate(pairs):
        # Keep solids visible while focusing on a single contact pair.
        visible = [i in always_on for i in range(n_traces)]
        for idx in contact_pair_to_indices[i]:
            if 0 <= idx < n_traces:
                visible[idx] = True
        label = f"{i + 1:02d}: {pair.master} / {pair.slave}"
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": args.title + f" - {pair.master} / {pair.slave}",
                        **_scene_range_layout(pair_bounds[i]),
                    },
                ],
            }
        )

    # Animation UI
    play_button = {
        "label": "Play",
        "method": "animate",
        "args": [
            None,
            {
                "frame": {"duration": int(1000 / max(args.fps, 1)), "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
            },
        ],
    }
    pause_button = {
        "label": "Pause",
        "method": "animate",
        "args": [
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
    }

    sliders = []
    if frames:
        steps = []
        for fr in frames:
            steps.append(
                {
                    "label": fr.name,
                    "method": "animate",
                    "args": [
                        [fr.name],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                }
            )
        sliders = [
            {
                "active": 0,
                "pad": {"t": 30},
                "steps": steps,
            }
        ]

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=args.title + " - All Contact Pairs",
        template="plotly_white",
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
            "xaxis": {"visible": False, "range": [float(global_bounds[0][0]), float(global_bounds[1][0])]},
            "yaxis": {"visible": False, "range": [float(global_bounds[0][1]), float(global_bounds[1][1])]},
            "zaxis": {"visible": False, "range": [float(global_bounds[0][2]), float(global_bounds[1][2])]},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 0.99,
                "showactive": True,
                "buttons": buttons,
            },
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.01,
                "y": 0.9,
                "buttons": [play_button, pause_button],
            },
        ],
        sliders=sliders,
        legend={"orientation": "h", "y": -0.05, "groupclick": "togglegroup"},
        uirevision="keep",
        annotations=[
            {
                "text": "Dropdown: select contact pair | Play: tightening animation | Drag to rotate | Panel: clip/ghost/selection",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.02,
                "showarrow": False,
                "font": {"size": 12},
            }
        ],
    )

    div_id = "contact-viewer"
    clip_bounds = {
        "x": [float(global_bounds[0][0]), float(global_bounds[1][0])],
        "y": [float(global_bounds[0][1]), float(global_bounds[1][1])],
        "z": [float(global_bounds[0][2]), float(global_bounds[1][2])],
    }
    group_indices = {
        "base": base_indices,
        "bolt": bolt_indices,
        "nut": nut_indices,
        "contact": contact_indices,
    }
    default_opacity = {
        "base": float(args.solid_alpha),
        "bolt": float(args.bolt_alpha),
        "nut": float(args.nut_alpha),
        "contact": float(args.contact_alpha),
    }
    post_script = f"""
(function() {{
  var gd = document.getElementById('{div_id}');
  if (!gd) {{
    return;
  }}
  var baseRanges = {json.dumps(clip_bounds)};
  var groupIndices = {json.dumps(group_indices)};
  var defaultOpacity = {json.dumps(default_opacity)};
  var ghostAlpha = 0.1;
  var isClipUpdate = false;
  var ghostOn = false;
  var originalColors = gd.data.map(function(t) {{ return t.color; }});
  var selectedIdx = null;

  var parent = gd.parentNode || document.body;
  if (parent && parent.style) {{
    parent.style.position = 'relative';
  }}
  var panel = document.createElement('div');
  panel.id = 'viewer-panel';
  panel.style.cssText = 'position:absolute;top:10px;right:10px;z-index:20;background:rgba(255,255,255,0.92);padding:8px 10px;border-radius:6px;font-family:Arial, sans-serif;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,0.2);max-width:220px;';
  panel.innerHTML = ''
    + '<div style=\"font-weight:bold;margin-bottom:6px;\">Viewer Controls</div>'
    + '<div style=\"margin-bottom:4px;\">Clip X (max)</div>'
    + '<input id=\"clip-x\" type=\"range\" min=\"1\" max=\"100\" value=\"100\" style=\"width:100%;\">'
    + '<div style=\"margin:6px 0 4px;\">Clip Y (max)</div>'
    + '<input id=\"clip-y\" type=\"range\" min=\"1\" max=\"100\" value=\"100\" style=\"width:100%;\">'
    + '<div style=\"margin:6px 0 4px;\">Clip Z (max)</div>'
    + '<input id=\"clip-z\" type=\"range\" min=\"1\" max=\"100\" value=\"100\" style=\"width:100%;\">'
    + '<div style=\"margin-top:6px;\"><button id=\"clip-reset\" style=\"width:100%;\">Reset Clip</button></div>'
    + '<div style=\"margin-top:6px;\"><button id=\"ghost-toggle\" style=\"width:100%;\">Ghost: Off</button></div>'
    + '<div id=\"selected-part\" style=\"margin-top:6px;\">Selected: none</div>';
  parent.appendChild(panel);

  var clipX = document.getElementById('clip-x');
  var clipY = document.getElementById('clip-y');
  var clipZ = document.getElementById('clip-z');
  var clipReset = document.getElementById('clip-reset');
  var ghostToggle = document.getElementById('ghost-toggle');
  var selectedLabel = document.getElementById('selected-part');

  function applyClip() {{
    var fx = Math.max(0.01, parseFloat(clipX.value) / 100.0);
    var fy = Math.max(0.01, parseFloat(clipY.value) / 100.0);
    var fz = Math.max(0.01, parseFloat(clipZ.value) / 100.0);
    var xr = baseRanges.x;
    var yr = baseRanges.y;
    var zr = baseRanges.z;
    var x1 = xr[0] + (xr[1] - xr[0]) * fx;
    var y1 = yr[0] + (yr[1] - yr[0]) * fy;
    var z1 = zr[0] + (zr[1] - zr[0]) * fz;
    isClipUpdate = true;
    Plotly.relayout(gd, {{
      'scene.xaxis.range': [xr[0], x1],
      'scene.yaxis.range': [yr[0], y1],
      'scene.zaxis.range': [zr[0], z1],
    }}).then(function() {{
      isClipUpdate = false;
    }});
  }}

  function setGroupOpacity(indices, value) {{
    if (!indices || indices.length === 0) {{
      return;
    }}
    Plotly.restyle(gd, {{'opacity': value}}, indices);
  }}

  function updateGhost() {{
    if (ghostOn) {{
      setGroupOpacity(groupIndices.base, ghostAlpha);
      setGroupOpacity(groupIndices.bolt, ghostAlpha);
      setGroupOpacity(groupIndices.nut, ghostAlpha);
      setGroupOpacity(groupIndices.contact, defaultOpacity.contact);
      ghostToggle.textContent = 'Ghost: On';
    }} else {{
      setGroupOpacity(groupIndices.base, defaultOpacity.base);
      setGroupOpacity(groupIndices.bolt, defaultOpacity.bolt);
      setGroupOpacity(groupIndices.nut, defaultOpacity.nut);
      setGroupOpacity(groupIndices.contact, defaultOpacity.contact);
      ghostToggle.textContent = 'Ghost: Off';
    }}
  }}

  clipX.addEventListener('input', applyClip);
  clipY.addEventListener('input', applyClip);
  clipZ.addEventListener('input', applyClip);
  clipReset.addEventListener('click', function() {{
    clipX.value = 100;
    clipY.value = 100;
    clipZ.value = 100;
    applyClip();
  }});
  ghostToggle.addEventListener('click', function() {{
    ghostOn = !ghostOn;
    updateGhost();
  }});

  gd.on('plotly_click', function(ev) {{
    if (!ev || !ev.points || ev.points.length === 0) {{
      return;
    }}
    var idx = ev.points[0].curveNumber;
    if (selectedIdx !== null) {{
      Plotly.restyle(gd, {{'color': originalColors[selectedIdx]}}, [selectedIdx]);
    }}
    if (selectedIdx === idx) {{
      selectedIdx = null;
      selectedLabel.textContent = 'Selected: none';
      return;
    }}
    selectedIdx = idx;
    Plotly.restyle(gd, {{'color': 'rgb(255,215,0)'}}, [idx]);
    selectedLabel.textContent = 'Selected: ' + (gd.data[idx].name || idx);
  }});

  gd.on('plotly_relayout', function(ev) {{
    if (isClipUpdate) {{
      return;
    }}
    var updated = false;
    if (ev['scene.xaxis.range'] !== undefined) {{
      baseRanges.x = ev['scene.xaxis.range'];
      updated = true;
    }}
    if (ev['scene.yaxis.range'] !== undefined) {{
      baseRanges.y = ev['scene.yaxis.range'];
      updated = true;
    }}
    if (ev['scene.zaxis.range'] !== undefined) {{
      baseRanges.z = ev['scene.zaxis.range'];
      updated = true;
    }}
    if (ev['scene.xaxis.range[0]'] !== undefined && ev['scene.xaxis.range[1]'] !== undefined) {{
      baseRanges.x = [ev['scene.xaxis.range[0]'], ev['scene.xaxis.range[1]']];
      updated = true;
    }}
    if (ev['scene.yaxis.range[0]'] !== undefined && ev['scene.yaxis.range[1]'] !== undefined) {{
      baseRanges.y = [ev['scene.yaxis.range[0]'], ev['scene.yaxis.range[1]']];
      updated = true;
    }}
    if (ev['scene.zaxis.range[0]'] !== undefined && ev['scene.zaxis.range[1]'] !== undefined) {{
      baseRanges.z = [ev['scene.zaxis.range[0]'], ev['scene.zaxis.range[1]']];
      updated = true;
    }}
    if (updated) {{
      clipX.value = 100;
      clipY.value = 100;
      clipZ.value = 100;
    }}
  }});
}})();
"""

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    include_js = "cdn" if args.cdn else "inline"
    pio.write_html(
        fig,
        args.out_html,
        include_plotlyjs=include_js,
        full_html=True,
        config={"responsive": True, "displayModeBar": True},
        div_id=div_id,
        post_script=post_script,
    )

    meta = {
        "mesh_path": mesh_path,
        "nuts": [
            {
                "name": nut["name"],
                "part": nut["part"],
                "axis": nut["axis"].tolist(),
                "center": nut["center"].tolist(),
            }
            for nut in nut_meshes
        ],
        "contact_pairs": [
            {"index": i + 1, "master": pair.master, "slave": pair.slave} for i, pair in enumerate(pairs)
        ],
    }
    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] html saved: {args.out_html}")
    print(f"[ok] meta saved: {args.out_meta}")


if __name__ == "__main__":
    main()
