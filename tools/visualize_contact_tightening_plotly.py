#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Plotly HTML viewer for solids, contact pairs, and nut tightening.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _mesh_trace(
    name: str,
    verts: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int],
    opacity: float,
    showlegend: bool = False,
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
        flatshading=True,
        lighting={"ambient": 0.3, "diffuse": 0.8, "specular": 0.2, "roughness": 0.6},
        showscale=False,
        showlegend=showlegend,
        hoverinfo="name",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plotly HTML viewer for contact pairs and tightening.")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--out-html", default=os.path.join(ROOT, "results", "contact_viewer.html"))
    ap.add_argument("--out-meta", default=os.path.join(ROOT, "results", "contact_viewer_meta.json"))
    ap.add_argument("--max-pairs", type=int, default=0, help="Limit contact pairs (0=all).")
    ap.add_argument("--max-tris", type=int, default=2500, help="Max triangles per contact surface.")
    ap.add_argument("--solid-max-tris", type=int, default=6000, help="Max triangles per solid part.")
    ap.add_argument("--frames-per-nut", type=int, default=8)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--contact-alpha", type=float, default=0.95)
    ap.add_argument("--solid-alpha", type=float, default=0.3)
    ap.add_argument("--bolt-alpha", type=float, default=0.8)
    ap.add_argument("--nut-alpha", type=float, default=1.0)
    ap.add_argument("--slave-lighten", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="Contact Viewer")
    ap.add_argument("--inline", action="store_true", help="Inline plotly.js (offline use).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    mesh_path = _resolve_mesh_path(cfg, args.config)
    asm = _load_asm(mesh_path)

    rng = np.random.default_rng(args.seed)

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
        part_mesh_by_name[part_name] = {
            "name": part_name,
            "verts": verts,
            "faces": faces,
        }

    base_meshes = [
        part_mesh_by_name[name]
        for name in sorted(part_mesh_by_name.keys())
        if name not in nut_parts and name not in bolt_parts
    ]
    bolt_meshes = [part_mesh_by_name[name] for name in sorted(bolt_parts) if name in part_mesh_by_name]

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
            verts, faces = _tri_surface_to_mesh(part, ts.tri_node_ids, args.max_tris, rng)
            if verts.size == 0 or faces.size == 0:
                continue
            contact_mesh_by_name[key] = {
                "name": key,
                "part": ts.part_name,
                "verts": verts,
                "faces": faces,
            }

    # Build traces
    traces: List[go.Mesh3d] = []
    trace_names: List[str] = []
    base_indices: List[int] = []
    bolt_indices: List[int] = []
    nut_indices: List[int] = []
    contact_indices: List[int] = []
    contact_pair_to_indices: List[List[int]] = []

    solid_color = (200, 200, 200)
    bolt_color = (120, 120, 120)
    color_cycle = _color_cycle()

    for mesh in base_meshes:
        trace = _mesh_trace(mesh["name"], mesh["verts"], mesh["faces"], solid_color, args.solid_alpha)
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(mesh["name"])
        base_indices.append(len(traces) - 1)

    for mesh in bolt_meshes:
        trace = _mesh_trace(mesh["name"], mesh["verts"], mesh["faces"], bolt_color, args.bolt_alpha)
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(mesh["name"])
        bolt_indices.append(len(traces) - 1)

    a_min, a_max, unit, clockwise = _tighten_angles_from_cfg(cfg)
    unit_rad = unit.startswith("deg")
    nut_colors = color_cycle
    for i, nut in enumerate(nut_meshes):
        color = nut_colors[i % len(nut_colors)]
        angle = a_min
        if unit_rad:
            angle = np.deg2rad(angle)
        angle = -angle if clockwise else angle
        v0 = _rotate_points(nut["verts"], nut["axis"], nut["center"], float(angle))
        trace = _mesh_trace(nut["name"], v0, nut["faces"], color, args.nut_alpha, showlegend=True)
        if trace is None:
            continue
        traces.append(trace)
        trace_names.append(nut["name"])
        nut_indices.append(len(traces) - 1)

    for i, pair in enumerate(pairs):
        pair_indices: List[int] = []
        pair_color = color_cycle[i % len(color_cycle)]
        for role, key in (("master", pair.master), ("slave", pair.slave)):
            mesh = contact_mesh_by_name.get(key)
            if mesh is None:
                continue
            color = pair_color if role == "master" else _lighten(pair_color, args.slave_lighten)
            trace = _mesh_trace(
                f"{key} ({role})",
                mesh["verts"],
                mesh["faces"],
                color,
                args.contact_alpha,
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

    buttons = [
        {
            "label": "All Pairs",
            "method": "update",
            "args": [
                {"visible": all_contacts_visible},
                {"title": args.title + " - All Contact Pairs"},
            ],
        }
    ]

    for i, pair in enumerate(pairs):
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
                    {"title": args.title + f" - {pair.master} / {pair.slave}"},
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
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.0,
                "y": 1.15,
                "showactive": True,
                "buttons": buttons,
            },
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.05,
                "buttons": [play_button, pause_button],
            },
        ],
        sliders=sliders,
        legend={"orientation": "h", "y": -0.05},
    )

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    include_js = "inline" if args.inline else "cdn"
    pio.write_html(fig, args.out_html, include_plotlyjs=include_js, full_html=True)

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
