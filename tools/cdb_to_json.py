#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ANSYS CDB to a compact JSON representation.

Goals:
  - Preserve all data values (nodes, elements, components, constraints, etc.).
  - Reduce size by removing fixed-width padding and using compact JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


def _parse_fixed_width(line: str, widths: List[int]) -> List[str]:
    out: List[str] = []
    idx = 0
    line = line.rstrip("\n")
    total = sum(widths)
    if len(line) < total:
        line = line + (" " * (total - len(line)))
    for w in widths:
        out.append(line[idx : idx + w].strip())
        idx += w
    return out


def _safe_int(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _safe_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _expand_range_stream(values: List[int]) -> List[int]:
    out: List[int] = []
    last = None
    for v in values:
        if v is None:
            continue
        if v < 0 and last is not None:
            out.extend(range(last, abs(v) + 1))
        else:
            out.append(v)
            last = v
    return out


def _parse_cdb(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    etblock: Dict[str, object] = {}
    nblock: Dict[str, object] = {}
    eblock: Dict[str, object] = {}
    cmblocks: List[Dict[str, object]] = []
    d_lines: List[str] = []
    other_lines: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith("ETBLOCK"):
            fmt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            i += 2
            data: List[Optional[int]] = []
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("-1"):
                    i += 1
                    break
                parts = _parse_fixed_width(lines[i], [9, 9])
                data.extend([_safe_int(parts[0]), _safe_int(parts[1])])
                i += 1
            etblock = {
                "format": fmt,
                "stride": 2,
                "data": data,
            }
            continue

        if line.startswith("NBLOCK"):
            fmt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            i += 2
            int_widths = [9, 9, 9]
            float_widths = [21] * 6
            ints: List[Optional[int]] = []
            floats: List[Optional[float]] = []
            count = 0
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("N,") or s.startswith("EBLOCK") or s.startswith("CMBLOCK") or s.startswith("-1"):
                    if s.startswith("-1"):
                        i += 1
                    break
                head = _parse_fixed_width(lines[i], int_widths)
                tail = _parse_fixed_width(lines[i][sum(int_widths) :], float_widths)
                ints.extend([_safe_int(v) for v in head])
                floats.extend([_safe_float(v) for v in tail])
                count += 1
                i += 1
            nblock = {
                "format": fmt,
                "int_stride": 3,
                "float_stride": 6,
                "count": count,
                "ints": ints,
                "floats": floats,
            }
            continue

        if line.startswith("EBLOCK"):
            fmt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            i += 2
            widths = [10] * 19
            data: List[Optional[int]] = []
            count = 0
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("-1"):
                    i += 1
                    break
                if s.startswith("CMBLOCK") or s.startswith("RLBLOCK"):
                    break
                fields = _parse_fixed_width(lines[i], widths)
                data.extend([_safe_int(v) for v in fields])
                count += 1
                i += 1
            eblock = {
                "format": fmt,
                "stride": 19,
                "count": count,
                "data": data,
            }
            continue

        if line.startswith("CMBLOCK"):
            header = line.strip()
            parts = [p.strip() for p in header.split(",")]
            name = parts[1] if len(parts) > 1 else f"CMBLOCK_{i}"
            ctype = parts[2] if len(parts) > 2 else ""
            raw_count = _safe_int(parts[3]) if len(parts) > 3 else None
            i += 1
            fmt = ""
            if i < len(lines) and lines[i].strip().startswith("("):
                fmt = lines[i].strip()
                i += 1
            data_vals: List[int] = []
            widths = [10] * 8
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("CMBLOCK") or s.startswith("RLBLOCK") or s.startswith("NBLOCK") or s.startswith("EBLOCK"):
                    break
                if s.startswith("-1"):
                    i += 1
                    break
                fields = _parse_fixed_width(lines[i], widths)
                ints = [_safe_int(x) for x in fields]
                for v in ints:
                    if v is not None:
                        data_vals.append(int(v))
                i += 1
            ids = _expand_range_stream(data_vals)
            cmblocks.append(
                {
                    "name": name,
                    "ctype": ctype,
                    "count_header": raw_count,
                    "format": fmt,
                    "ids": ids,
                }
            )
            continue

        if line.startswith("D,"):
            d_lines.append(line.strip())
            i += 1
            continue

        other_lines.append(line.rstrip("\n"))
        i += 1

    return {
        "meta": {
            "source": os.path.basename(path),
            "source_bytes": os.path.getsize(path),
            "line_count": len(lines),
        },
        "etblock": etblock,
        "nblock": nblock,
        "eblock": eblock,
        "cmblock": cmblocks,
        "d": d_lines,
        "other_lines": other_lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert ANSYS CDB to compact JSON.")
    ap.add_argument("--cdb", required=True, help="Path to .cdb file")
    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: results/<cdb_basename>.json)",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (larger).")
    args = ap.parse_args()

    cdb_path = os.path.abspath(args.cdb)
    if not os.path.exists(cdb_path):
        raise FileNotFoundError(f"CDB not found: {cdb_path}")

    data = _parse_cdb(cdb_path)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_dir = os.path.join(root, "results")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(cdb_path))[0]
        out_path = os.path.join(out_dir, f"{base}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(data, f, indent=2, ensure_ascii=True)
        else:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=True)

    print(f"[ok] JSON saved: {out_path}")


if __name__ == "__main__":
    main()
