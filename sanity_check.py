# -*- coding: utf-8 -*-
"""
sanity_check.py — 训练前快速体检（PyCharm 友好版，仅 tqdm.auto）
检查项：
  [1] 载入 & 构建（仅 build，不训练）
  [2] 体积分点 + 材料映射（严格形状/覆盖校验）
  [3] 预紧采样（螺栓个数与上下采样点统计）
  [4] 接触识别（从 INP 打印 Contact Pair / Tie / 摩擦系数；并打印每个接触表面详情 + MIRROR UP 表面详情）
退出码：
  0 = 通过；1 = build 失败；2 = 体积分点缺失/异常；3 = 其它非致命问题
"""

from __future__ import annotations
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, List

# 让 src 可导入
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tqdm.auto import tqdm

# 兼容两种导入路径
try:
    from train.trainer import Trainer, TrainerConfig  # type: ignore
except Exception:
    from src.train.trainer import Trainer, TrainerConfig  # type: ignore


# ---------- 工具：扫描 INP 里的 Interaction 与摩擦系数 ----------
def _scan_interactions(inp_path: str) -> Dict[str, Dict[str, Any]]:
    """
    扫描 INP 中的 *Surface Interaction / *Contact Property ，提取 interaction 名与摩擦系数。
    返回 dict: { interaction_name : { 'mu': float|None, 'source': 'surface'|'property' } }
    """
    info: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(inp_path):
        return info

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    cur_name = None
    cur_kind = None  # 'surface' or 'property'
    in_friction = False

    def _commit_mu(mu_val: Optional[float]):
        if cur_name:
            rec = info.setdefault(cur_name, {'mu': None, 'source': cur_kind})
            if rec.get('mu') is None and (mu_val is not None):
                rec['mu'] = mu_val

    def _parse_mu_from_line(s: str) -> Optional[float]:
        try:
            toks = [t.strip() for t in s.split(",") if t.strip()]
            if not toks:
                return None
            return float(toks[0])
        except Exception:
            return None

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        i += 1
        if not s or s.startswith("**"):
            continue
        s_up = s.upper()

        if s_up.startswith("*SURFACE INTERACTION"):
            cur_name, cur_kind = None, 'surface'
            in_friction = False
            import re
            m = re.search(r"name\s*=\s*([^,]+)", s, flags=re.IGNORECASE)
            if m:
                cur_name = m.group(1).strip().strip('"').strip("'")
                info.setdefault(cur_name, {'mu': None, 'source': 'surface'})
            continue

        if s_up.startswith("*CONTACT PROPERTY"):
            cur_name, cur_kind = None, 'property'
            in_friction = False
            import re
            m = re.search(r"name\s*=\s*([^,]+)", s, flags=re.IGNORECASE)
            if m:
                cur_name = m.group(1).strip().strip('"').strip("'")
                info.setdefault(cur_name, {'mu': None, 'source': 'property'})
            continue

        if s_up.startswith("*FRICTION"):
            in_friction = True
            continue

        if s_up.startswith("*"):
            in_friction = False
            continue

        if in_friction and cur_name:
            mu = _parse_mu_from_line(s)
            if mu is not None:
                _commit_mu(mu)
                in_friction = False

    return info


def _count_inp_keywords(inp_path: str) -> Dict[str, int]:
    keys = {
        "CONTACT_PAIR": 0,
        "CONTACT": 0,
        "TIE": 0,
        "SURFACE": 0,
        "SURFACE_INTERACTION": 0,
        "CONTACT_PROPERTY": 0,
        "ELEMENT_C3D": 0,
        "ELEMENT_SHELL": 0,
    }
    try:
        with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip().upper()
                if s.startswith("*CONTACT PAIR"):
                    keys["CONTACT_PAIR"] += 1
                elif s.startswith("*SURFACE INTERACTION"):
                    keys["SURFACE_INTERACTION"] += 1
                elif s.startswith("*CONTACT PROPERTY"):
                    keys["CONTACT_PROPERTY"] += 1
                elif s.startswith("*TIE"):
                    keys["TIE"] += 1
                elif s.startswith("*CONTACT"):
                    keys["CONTACT"] += 1
                elif s.startswith("*SURFACE"):
                    keys["SURFACE"] += 1
                elif s.startswith("*ELEMENT") and "TYPE=C3D" in s:
                    keys["ELEMENT_C3D"] += 1
                elif s.startswith("*ELEMENT") and ("TYPE=S" in s or "TYPE=SC" in s):
                    keys["ELEMENT_SHELL"] += 1
    except Exception:
        pass
    return keys


def _shape_of(x: Any) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(getattr(x, "shape", None)) if x is not None else None
    except Exception:
        return None


def _use_config_from_main_or_default() -> TrainerConfig:
    """
    优先调用 main._prepare_config_with_autoguess()；若不可用则回退到默认 TrainerConfig()
    并打印材料与映射信息。
    """
    print("======================================================================")
    print("[1/4 载入 & 构建]")
    print("======================================================================")
    print("[info] 使用 main._prepare_config_with_autoguess() 生成配置。")
    cfg: TrainerConfig
    try:
        import main  # 你的项目 main.py
        if hasattr(main, "_prepare_config_with_autoguess"):
            cfg = main._prepare_config_with_autoguess()  # type: ignore
        else:
            raise AttributeError
    except Exception:
        cfg = TrainerConfig()
        print("[warn] 无法调用 main._prepare_config_with_autoguess()，使用 TrainerConfig() 默认值。")

    inp = os.path.abspath(cfg.inp_path)
    print(f"[OK] INP: {inp}")
    print("[info] 材料表（name -> (E, nu) 或属性）")
    for k, v in (cfg.materials or {}).items():
        print(f"  {k}: {v}")
    print("[info] part2mat（零件名 -> 材料名）")
    for k, v in (cfg.part2mat or {}).items():
        print(f"  {k}: {v}")

    if os.path.isfile(inp):
        stats = _count_inp_keywords(inp)
        print("[info] INP 粗略关键字统计：")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        try:
            from inp_contacts import quick_summary  # type: ignore
            print("\n" + quick_summary(inp))
        except Exception:
            pass
    return cfg


# ---------- 表面匹配 & 打印工具 ----------
def _dequote(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _surface_key_variants(raw: str) -> List[str]:
    """
    生成一组可能的 surface key 变体，容错：是否带 ASM:: 前缀、是否带引号、大小写/空格。
    """
    base = _dequote(raw)
    cands = []

    # 原样/去引号/加引号
    cands += [raw, base, f'"{base}"']

    # 加入 ASM:: 前缀的版本
    cands += [f'ASM::{raw}', f'ASM::{base}', f'ASM::"{base}"']

    # 大小写宽松与去空格
    low = base.lower(); up = base.upper(); nos = base.replace(" ", "")
    cands += [low, up, f'"{low}"', f'"{up}"', nos, f'"{nos}"',
              f'ASM::{low}', f'ASM::"{low}"', f'ASM::{up}', f'ASM::"{up}"',
              f'ASM::{nos}', f'ASM::"{nos}"']

    # 去重保持顺序
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _lookup_surface_key(asm, name_like: str) -> Optional[str]:
    """
    在 asm.surfaces 中查找最可能匹配 name_like 的 key。
    """
    if not asm or not getattr(asm, "surfaces", None):
        return None
    # 先直接命中
    if name_like in asm.surfaces:
        return name_like
    # 再试变体
    for k in _surface_key_variants(name_like):
        if k in asm.surfaces:
            return k
    # 最后做一次“相似”查找（忽略大小写与空格）
    want = _dequote(name_like).replace(" ", "").lower()
    for k in asm.surfaces.keys():
        kk = _dequote(k).replace(" ", "").lower()
        if (want in kk) or (kk in want):
            return k
    return None


def _print_surface_details(asm, key: str, title: str = "") -> None:
    """
    打印一个表面的详情：scope/type/owner、items 样例、元素面统计、首个面顶点坐标。
    """
    try:
        sdef = asm.surfaces[key]
    except KeyError:
        print(f"  [SURFACE] {title} 未找到：{key}")
        return

    scope = getattr(sdef, "scope", "?")
    stype = getattr(sdef, "stype", "?")
    owner = getattr(sdef, "owner", None)
    items = list(getattr(sdef, "items", []))
    print(f"  [SURFACE] {title}: key='{key}', scope={scope}, type={stype}, owner={owner}, items={len(items)}")
    for i, it in enumerate(items[:3]):
        print(f"    [items[{i}]] {it}")

    # 统计覆盖的元素面数量，并尝试打印首个面的四点坐标
    face_total = 0
    first_coords_printed = False
    for (elset_name, sface) in items:
        try:
            eids = asm.expand_elset(elset_name)
        except Exception as e:
            print(f"    [warn] expand_elset('{elset_name}') 失败：{e}")
            continue
        face_total += len(eids)
        if (not first_coords_printed) and eids:
            try:
                if sface and sface.upper().startswith("S"):
                    fid = int(sface[1:])
                else:
                    fid = None
                if fid is not None:
                    coords = asm.get_face_nodes(int(eids[0]), fid)  # (4,3) for C3D8
                    print(f"    [example] 第一个元素面 eid={eids[0]}, {sface} 顶点坐标(4x3)：")
                    for r in coords:
                        print(f"      ({r[0]:.6g}, {r[1]:.6g}, {r[2]:.6g})")
                    first_coords_printed = True
            except Exception as e:
                print(f"    [warn] 读取第一个元素面的坐标失败：{e}")

    print(f"    [stats] 元素面计数（按 ELSET×面号展开）：{face_total}")


def _print_pair_surfaces(asm, master_name: str, slave_name: str) -> None:
    mkey = _lookup_surface_key(asm, master_name)
    skey = _lookup_surface_key(asm, slave_name)
    if mkey:
        _print_surface_details(asm, mkey, title="master")
    else:
        print(f"  [SURFACE] master 未找到匹配：{master_name}")
    if skey:
        _print_surface_details(asm, skey, title="slave")
    else:
        print(f"  [SURFACE] slave 未找到匹配：{slave_name}")


# ---------- 主流程 ----------
def main():
    # [1] 配置与构建
    cfg = _use_config_from_main_or_default()

    # 预紧链路简要调试
    print("\n[DEBUG] 预紧表面链路检查（SurfaceDef → ELSET → 元素面顶点）")
    try:
        from inp_io.inp_parser import load_inp
        asm = load_inp(cfg.inp_path)

        def _inspect_surface(key: str, tag: str):
            if key not in asm.surfaces:
                print(f"[DEBUG] [{tag}] 未在 asm.surfaces 中找到：{key}")
                return
            sdef = asm.surfaces[key]
            print(f"[DEBUG] [{tag}] key='{key}', stype={getattr(sdef,'stype','?')}, items={len(getattr(sdef,'items',[]))}")
            items = list(getattr(sdef, "items", []))[:1]
            if items:
                print(f"    [DEBUG] items 样例: {items[0]}")

        pls = getattr(cfg, "preload_specs", []) or []
        if not pls:
            print("[DEBUG] cfg.preload_specs 为空；请确认 main.py 中的 BOLT_SURFACES 是否已填写。")
        else:
            for i, sp in enumerate(pls):
                upk = sp.get("up_key") or sp.get("up") or ""
                dnk = sp.get("down_key") or sp.get("down") or ""
                print(f"[DEBUG] ---- bolt[{i}] name={sp.get('name','?')} ----")
                _inspect_surface(upk, f"bolt[{i}].UP")
                _inspect_surface(dnk, f"bolt[{i}].DN")
    except Exception as e:
        print(f"[DEBUG] 表面链路调试块异常：{e}")

    trainer = Trainer(cfg)
    try:
        with tqdm(total=1, desc="Build", leave=True) as pbar:
            trainer.build()
            pbar.update(1)
        print("[OK] 调用 trainer.build() 成功。")
    except Exception:
        print("[ERR] trainer.build() 失败，堆栈如下：")
        traceback.print_exc()
        sys.exit(1)

    methods = [m for m in dir(trainer) if not m.startswith("_")]
    print("\n[hint] Trainer 中可用的公共方法（节选）：")
    for name in methods:
        if name in ("build", "run"):
            print(f"  - {name}")

    # [2] 体积分点 + 材料映射
    print("\n======================================================================")
    print("[2/4 体积分点 + 材料映射]")
    print("======================================================================")

    X_vol = getattr(trainer, "X_vol", None)
    w_vol = getattr(trainer, "w_vol", None)
    mat_id = getattr(trainer, "mat_id", None)
    if X_vol is None or w_vol is None or mat_id is None:
        elas = getattr(trainer, "elasticity", None)
        if elas is not None:
            X_vol = getattr(elas, "X_vol", None)
            w_vol = getattr(elas, "w_vol", None)
            mat_id = getattr(elas, "mat_id", None)

    def _shape(x):
        try:
            return tuple(getattr(x, "shape", None)) if x is not None else None
        except Exception:
            return None

    print(f"[volume] X_vol: {_shape(X_vol)}, w_vol: {_shape(w_vol)}, mat_id: {_shape(mat_id)}")

    bad = False
    if X_vol is None or w_vol is None or mat_id is None:
        bad = True
    else:
        nx = getattr(X_vol, "shape", [0])[0]
        nw = getattr(w_vol, "shape", [0])[0]
        nm = getattr(mat_id, "shape", [0])[0]
        if nx == 0 or nx != nw or nx != nm:
            bad = True

    if bad:
        print("[ERR] 体积分点未构建成功（X_vol 或 w_vol 为 None，或长度不一致/为 0）。")
        print("      排查建议：")
        print("      1) INP 中是否存在用于体积分的 **实体单元(C3D*)**；若只有壳(S*)，需改用实体网格或在采样中实现壳回退；")
        print("      2) build_volume_points/_build_volume_points 是否被正确调用；")
        print("      3) part2mat 与 INP 零件名是否一一匹配（大小写/空格/后缀）；")
        print("      4) 坐标缩放/掩膜是否把点过滤为空；")
        print("      5) 运行 Training 也会很快在同处报错，可定位缺项根源。")
        sys.exit(2)

    try:
        uniq_ids = sorted(list(set(int(i) for i in mat_id.tolist())))
    except Exception:
        uniq_ids = []
    print(f"[ok] 体积分点中的材料 id 出现：{uniq_ids}")

    id2props = getattr(trainer, "id2props_map", None)
    if isinstance(id2props, dict) and id2props:
        print("[ok] id -> (E, nu) 映射：")
        enum_names = getattr(trainer, "enum_names", [])
        for i in uniq_ids:
            props = id2props.get(i)
            name = enum_names[i] if (isinstance(enum_names, list) and i < len(enum_names)) else "?"
            if props is None:
                print(f"  id {i}: [缺失]（请检查 part2mat 与 materials 对应关系）")
            else:
                if isinstance(props, (list, tuple)) and len(props) >= 2:
                    E, nu = float(props[0]), float(props[1])
                    print(f"  id {i}: name={name} props=({E:.6g}, {nu:.5g})")
                else:
                    print(f"  id {i}: name={name} props={props}")

    # [3] 预紧采样
    print("\n======================================================================")
    print("[3/4 预紧采样统计]")
    print("======================================================================")
    preload = getattr(trainer, "preload", None)
    if preload is None:
        print("[warn] 未构建预紧模型（preload=None）。若你期望有预紧，请检查 cfg.preload_specs 是否填入；键名需与 INP 完全一致。")
    else:
        bolts = getattr(preload, "bolts", None) or getattr(preload, "_bolts", None)
        if isinstance(bolts, list):
            print(f"[ok] 总螺栓数: {len(bolts)}")
            total_up = 0
            total_dn = 0
            for i, b in enumerate(bolts):
                name = getattr(b, "name", f"bolt{i+1}")

                def _pick_nonempty(*cands):
                    for arr in cands:
                        if arr is not None and getattr(arr, "size", 0) > 0:
                            return arr
                    return None

                X_up = _pick_nonempty(getattr(b, "X_up", None), getattr(b, "up_pts", None),
                                      getattr(b, "points_up", None))
                X_dn = _pick_nonempty(getattr(b, "X_dn", None), getattr(b, "dn_pts", None),
                                      getattr(b, "points_dn", None))
                n_up = 0 if X_up is None else getattr(X_up, "shape", [0])[0]
                n_dn = 0 if X_dn is None else getattr(X_dn, "shape", [0])[0]
                total_up += n_up
                total_dn += n_dn
                print(f"  bolt[{i}] name={name} up={n_up} dn={n_dn}")
            print(f"[ok] 预紧采样点总数：up={total_up}  dn={total_dn}  total={total_up+total_dn}")
        else:
            print("[warn] 预紧模型存在，但未能读出 bolts 列表（内部结构名不同）。")

    # [4] 接触识别与表面详情
    print("\n======================================================================")
    print("[4/4 接触对统计]")
    print("======================================================================")

    try:
        from inp_io.inp_parser import load_inp
        asm = load_inp(cfg.inp_path)
    except Exception as e:
        asm = None
        print(f"[warn] 读取 INP 以总结接触信息失败：{e}")

    # 4.1 打印 Contact Pair / Tie / Interaction（含 μ）
    if asm is not None:
        inter_mu = _scan_interactions(cfg.inp_path)

        if asm.contact_pairs:
            print(f"[INP] Contact Pair 共 {len(asm.contact_pairs)} 组：")
            for k, cp in enumerate(asm.contact_pairs, 1):
                inter = (cp.interaction or "").strip().strip('"').strip("'")
                mu_txt = ""
                if inter and inter in inter_mu:
                    mu_val = inter_mu[inter].get('mu', None)
                    if mu_val is not None:
                        mu_txt = f", μ={mu_val:g}"
                print(f"  - #{k}: master='{cp.master}', slave='{cp.slave}', interaction='{inter or 'N/A'}'{mu_txt}")
            print("\n[INP] 逐对接触表面详情：")
            for k, cp in enumerate(asm.contact_pairs, 1):
                print(f"  --- Pair #{k} ---")
                _print_pair_surfaces(asm, cp.master, cp.slave)
        else:
            print("[INP] 未在 *Contact Pair 中发现接触对。")

        if asm.ties:
            print(f"\n[INP] Tie（绑定）共 {len(asm.ties)} 组：")
            for k, t in enumerate(asm.ties, 1):
                print(f"  - #{k}: master='{t.master}', slave='{t.slave}'")
        else:
            print("\n[INP] 未在 *Tie 中发现绑定。")

        if inter_mu:
            print(f"\n[INP] Interaction 定义（含摩擦系数）共 {len(inter_mu)} 个：")
            for name, rec in inter_mu.items():
                mu = rec.get('mu', None)
                src = rec.get('source', '?')
                mu_s = f"{mu:g}" if (mu is not None) else "N/A"
                print(f"  - name='{name}', μ={mu_s}, source={src}")
        else:
            print("\n[INP] 未解析到 *Friction（摩擦系数）。")
    else:
        print("[INP] 略过 INP 接触摘要（asm=None）。")

    # 4.2 Trainer 内部 contact 概览
    contact = getattr(trainer, "contact", None)
    if contact is None:
        print("[trainer] 未构建接触模型（contact=None）。")
    else:
        n_nodes = getattr(contact, "n_nodes", None)
        n_pairs = getattr(contact, "n_pairs", None)
        info = []
        if n_pairs is not None:
            info.append(f"pairs={n_pairs}")
        if n_nodes is not None:
            info.append(f"nodes={n_nodes}")
        if info:
            print("[trainer] 已启用接触（" + ", ".join(info) + ")")

    # 4.3 额外打印目标表面：ASM::"MIRROR up"
    if asm is not None:
        print("\n[INP] 目标表面详情（用于变形云图）：MIRROR UP")
        target_names = ['ASM::"MIRROR up"', '"MIRROR up"', 'MIRROR up', 'mirror up']
        tkey = None
        for nm in target_names:
            tkey = _lookup_surface_key(asm, nm)
            if tkey:
                break
        if tkey:
            _print_surface_details(asm, tkey, title='MIRROR UP')
        else:
            print("  [SURFACE] 未找到 MIRROR UP（请检查名称大小写与是否在装配作用域 ASM::）。")

    print("\n✅ 体检完成。若有 [warn]/[ERR] 提示，请据此定位数据管线中的缺项或命名差异。")
    sys.exit(0)


if __name__ == "__main__":
    main()
