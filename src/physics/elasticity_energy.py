# -*- coding: utf-8 -*-
# src/physics/elasticity_energy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
import tensorflow as tf


@dataclass
class ElasticityConfig:
    """
    线弹性内能项的配置。
    - coord_scale:   坐标缩放（仅用于数值尺度；不会改变对外 X 的物理含义）
    - chunk_size:    分块大小（前向微批，降低峰值显存）
    - use_pfor:      仅为 batch_jacobian 兼容保留；有限差分不使用
    - check_nan:     是否在关键张量处进行数值检查
    - n_points_per_step: 每步用于 Jacobian/内能计算的体积分点上限（None=全量）
    """
    coord_scale: float = 1.0
    chunk_size: int = 64
    use_pfor: bool = False
    check_nan: bool = False
    n_points_per_step: Optional[int] = None


class ElasticityEnergy:
    r"""
    线弹性体的内能：
        E_int = ∫_Ω [ 1/2 * λ * (tr ε)^2 + μ * (ε : ε) ] dΩ,
    其中小应变张量
        ε = 0.5 * (∇u + ∇u^T)

    实现要点：
    - 材料参数 (λ, μ) 由 (E, ν) 推得，并按点展开到体积分点；
    - 默认用**有限差分列梯度**估计 J=∂u/∂x（只需一阶反传，避免二阶导引发的断链/OOM）；
    - 支持每步对子采样一部分体积分点（无偏估计）以稳显存；
    - 保留 batch_jacobian 的两条旧路径（需要时可切回）。
    """

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        X_vol: np.ndarray,     # (N,3) 体积分点坐标
        w_vol: np.ndarray,     # (N,)   对应积分权重/体积
        mat_id: np.ndarray,    # (N,)   材料 id（int）
        matlib: Any,           # 材料库（需能由 id/name 拿到 (E, ν)）
        cfg: ElasticityConfig,
    ) -> None:
        # 基本检查
        if X_vol is None or w_vol is None or mat_id is None:
            raise ValueError("[ElasticityEnergy] 需要 X_vol / w_vol / mat_id。")
        X_vol = np.asarray(X_vol)
        w_vol = np.asarray(w_vol)
        mat_id = np.asarray(mat_id)

        if X_vol.ndim != 2 or X_vol.shape[1] != 3:
            raise ValueError(f"[ElasticityEnergy] X_vol 形状应为 (N,3)，得到 {X_vol.shape}")
        if w_vol.ndim != 1 or mat_id.ndim != 1:
            raise ValueError(f"[ElasticityEnergy] w_vol 或 mat_id 维度错误：{w_vol.shape}, {mat_id.shape}")
        if not (len(X_vol) == len(w_vol) == len(mat_id)):
            raise ValueError("[ElasticityEnergy] X_vol / w_vol / mat_id 长度不一致")

        for name, arr in (("X_vol", X_vol), ("w_vol", w_vol), ("mat_id", mat_id)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"[ElasticityEnergy] 输入 {name} 含 NaN/Inf")

        # 缓存为 float32/int32
        self.X_np = X_vol.astype(np.float32, copy=False)
        self.w_np = w_vol.astype(np.float32, copy=False)
        self.mid_np = mat_id.astype(np.int32, copy=False)

        self.cfg = cfg

        # 材料参数：由材料库映射 id → (E, ν)，再转 λ、μ 并按点展开
        lam_tab, mu_tab = self._precompute_lame_params(self.mid_np, matlib)  # (K,), (K,)
        self.lam_np = lam_tab[self.mid_np]  # (N,)
        self.mu_np = mu_tab[self.mid_np]    # (N,)

        # 转为 Tensor（常量）
        self.X_tf = tf.convert_to_tensor(self.X_np)      # (N,3) float32
        self.w_tf = tf.convert_to_tensor(self.w_np)      # (N,)
        self.lam_tf = tf.convert_to_tensor(self.lam_np)  # (N,)
        self.mu_tf = tf.convert_to_tensor(self.mu_np)    # (N,)

        # 坐标缩放（用于差分步长的量纲调度；不会改变对外 X 的物理含义）
        self._scale = float(max(self.cfg.coord_scale, 1e-8))

    # ------------------------------------------------------------------ #
    # 顶层接口：计算内能（保持旧签名，不破坏外部调用）
    # ------------------------------------------------------------------ #
    def energy(
        self,
        u_fn,
        params: Optional[Dict[str, tf.Tensor]] = None,
        tape: Optional[tf.GradientTape] = None,  # 为兼容旧调用，有限差分路径中不使用
    ):
        """
        计算内能 E_int 与统计信息 stats。
        - 默认使用有限差分列梯度求 J，避免二阶导问题；
        - 支持 n_points_per_step 子采样，稳显存且无偏。
        """
        # 1) 选择本步参与计算的体积分点（无偏子采样）
        X_full, w_full = self.X_tf, self.w_tf
        lam_full, mu_full = self.lam_tf, self.mu_tf

        if self.cfg.n_points_per_step:
            n = tf.shape(X_full)[0]
            m = tf.minimum(tf.constant(int(self.cfg.n_points_per_step), tf.int32), n)
            idx = tf.random.shuffle(tf.range(n))[:m]

            X_use = tf.gather(X_full, idx)      # (M,3)
            w_use = tf.gather(w_full, idx)      # (M,)
            lam_use = tf.gather(lam_full, idx)  # (M,)
            mu_use = tf.gather(mu_full, idx)    # (M,)
        else:
            X_use, w_use, lam_use, mu_use = X_full, w_full, lam_full, mu_full

        # 2) 计算 Jacobian（默认：有限差分列梯度；如需切回 batch_jacobian，见下方注释）
        # J = self._batch_jacobian_chunked_with_tape(u_fn, params, tape, X_use)  # 旧式：需要外部持久 tape
        # J = self._batch_jacobian_chunked(u_fn, params, X_use)                  # 旧式：内部持久 tape + batch_jacobian
        J = self._jacobian_fd_columns(u_fn, params, X_use)                        # 新式：有限差分列梯度（推荐）

        # 3) 小应变 ε、能量密度 ψ 与积分
        eps = 0.5 * (J + tf.transpose(J, perm=[0, 2, 1]))  # (M,3,3)
        if self.cfg.check_nan:
            tf.debugging.check_numerics(eps, "strain tensor has NaN/Inf")

        tr_eps = tf.linalg.trace(eps)                       # (M,)
        eps_sq = tf.reduce_sum(eps * eps, axis=[1, 2])     # (M,)
        psi = 0.5 * lam_use * (tr_eps ** 2.0) + mu_use * eps_sq
        if self.cfg.check_nan:
            tf.debugging.check_numerics(psi, "energy density has NaN/Inf")

        E_int = tf.reduce_sum(w_use * psi)                 # 标量
        if self.cfg.check_nan:
            tf.debugging.check_numerics(E_int, "E_int has NaN/Inf")

        stats = {
            "N_total": int(self.X_np.shape[0]),
            "N_used": int(self.X_np.shape[0]) if self.cfg.n_points_per_step is None else int(psi.shape[0]),
            "chunk_size": int(self.cfg.chunk_size),
            "use_pfor": bool(self.cfg.use_pfor),
            "coord_scale": float(self._scale),
        }
        return E_int, stats

    # ------------------------------------------------------------------ #
    # 方案 A（推荐）：有限差分列梯度（只需一阶反传，避免二阶导）
    # ------------------------------------------------------------------ #
    def _jacobian_fd_columns(self, u_fn, params, X_tensor: tf.Tensor) -> tf.Tensor:
        """
        用中央差分近似 J = ∂u/∂x，按列构造：
          J[:, :, k] ≈ (u(X + h e_k) - u(X - h e_k)) / (2h)

        说明：
          - 完全由前向组成，外层 tape 能稳定捕捉到参数依赖（不需要二阶导）；
          - 配合 chunk 分块评估，显存可控；
          - h 的默认尺度与 coord_scale 相关，可按数值表现微调。
        """
        X_tensor = tf.cast(X_tensor, tf.float32)
        N = int(X_tensor.shape[0])
        chunk = int(max(1, self.cfg.chunk_size))

        # 差分步长：与坐标尺度自适应（经验起点：1e-3*scale）
        # 若数值抖动略大，可调成 5e-4*scale 或 2e-3*scale 试探
        h = tf.constant(1e-3 * max(self._scale, 1.0), dtype=tf.float32)

        cols = []
        for k in range(3):
            ek = tf.one_hot(k, 3, dtype=tf.float32)  # (3,)
            outs = []

            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                Xc = X_tensor[s:e]  # (m,3)

                Xp = Xc + h * ek
                Xm = Xc - h * ek

                Up = tf.cast(u_fn(Xp, params), tf.float32)  # (m,3)
                Um = tf.cast(u_fn(Xm, params), tf.float32)  # (m,3)

                if self.cfg.check_nan:
                    tf.debugging.check_numerics(Up, "u(X+h) NaN/Inf")
                    tf.debugging.check_numerics(Um, "u(X-h) NaN/Inf")

                col_k = (Up - Um) / (2.0 * h)  # (m,3)
                outs.append(col_k)

            col = tf.concat(outs, axis=0) if len(outs) > 1 else outs[0]  # (N,3)
            cols.append(col)

        # 组装 (N,3,3)，最后一维是列
        J = tf.stack(cols, axis=2)
        return J

    # ------------------------------------------------------------------ #
    # 方案 B（保留以兼容）：batch_jacobian + 外部持久 Tape
    # ------------------------------------------------------------------ #
    def _batch_jacobian_chunked_with_tape(
        self,
        u_fn,
        params,
        tape: Optional[tf.GradientTape],
        X_tensor: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        使用外部传入的持久 Tape 计算 batch_jacobian（旧实现）。
        - 若未传 X_tensor，则默认使用 self.X_tf（不建议：无法配合子采样）
        - 注意：该方式可能产生较高显存/二阶导开销，仅保留以兼容旧路径。
        """
        if tape is None:
            raise RuntimeError("[ElasticityEnergy] 需要传入有效的 GradientTape 才能使用 _batch_jacobian_chunked_with_tape。")

        Xsrc = self.X_tf if X_tensor is None else X_tensor
        N = int(Xsrc.shape[0])
        chunk = int(max(1, self.cfg.chunk_size))
        outs = []

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Xc = Xsrc[s:e]
            # 若沿用历史缩放，可改为 Xc_s = Xc / self._scale; Uc = u_fn(Xc_s * self._scale, params)
            tape.watch(Xc)
            Uc = u_fn(Xc, params)
            Uc = tf.cast(Uc, tf.float32)
            if self.cfg.use_pfor:
                J = tape.batch_jacobian(Uc, Xc, experimental_use_pfor=True)
            else:
                J = tape.batch_jacobian(Uc, Xc, experimental_use_pfor=False)
            outs.append(J)

        return tf.concat(outs, axis=0) if len(outs) > 1 else outs[0]

    # ------------------------------------------------------------------ #
    # 方案 C（保留以兼容）：batch_jacobian + 内部持久 Tape
    # ------------------------------------------------------------------ #
    def _batch_jacobian_chunked(
        self,
        u_fn,
        params,
        X_tensor: tf.Tensor,
    ) -> tf.Tensor:
        """
        内部开持久 Tape 并分块计算 batch_jacobian（旧实现）。
        - 与 _batch_jacobian_chunked_with_tape 等价，但不依赖外部 Tape；
        - 可能产生较高显存/二阶导开销，仅保留以兼容旧路径。
        """
        N = int(X_tensor.shape[0])
        chunk = int(max(1, self.cfg.chunk_size))
        outs = []

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Xc = tf.identity(X_tensor[s:e])
            with tf.GradientTape(persistent=True) as t:
                t.watch(Xc)
                Uc = u_fn(Xc, params)
                Uc = tf.cast(Uc, tf.float32)
                if self.cfg.check_nan:
                    tf.debugging.check_numerics(Uc, "u(X) has NaN/Inf")
            if self.cfg.use_pfor:
                J = t.batch_jacobian(Uc, Xc, experimental_use_pfor=True)
            else:
                J = t.batch_jacobian(Uc, Xc, experimental_use_pfor=False)
            outs.append(J)
            del t

        return tf.concat(outs, axis=0) if len(outs) > 1 else outs[0]

    # ------------------------------------------------------------------ #
    # 材料参数工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _E_nu_to_lame(E: float, nu: float) -> Tuple[float, float]:
        lam = float(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        mu  = float(E / (2.0 * (1.0 + nu)))
        return lam, mu

    def _precompute_lame_params(self, mat_id_np: np.ndarray, matlib: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        依据 materials 库推断 id→(E, ν)，并生成 id→(λ, μ) 的查表。
        兼容：
            - matlib.props_for_id(id) / name_for_id(id) / props_for_name(name)
            - matlib.id2name / matlib.materials / matlib.props / matlib.name2props
            - dict 形式：{id: (E,ν)} 或 {name: (E,ν)} + 'id2name'
        """
        uniq = np.unique(mat_id_np.astype(np.int32))
        max_id = int(uniq.max()) if uniq.size > 0 else 0
        lam_tab = np.zeros((max_id + 1,), dtype=np.float32)
        mu_tab  = np.zeros((max_id + 1,), dtype=np.float32)

        def _get_by_id(mid: int) -> Optional[Tuple[float, float]]:
            # 1) 直接 by id
            if hasattr(matlib, "props_for_id") and callable(getattr(matlib, "props_for_id")):
                try:
                    E, nu = matlib.props_for_id(mid)
                    return float(E), float(nu)
                except Exception:
                    pass
            # 2) 通过 name
            name = None
            if hasattr(matlib, "id2name"):
                try:
                    ref = getattr(matlib, "id2name")
                    name = ref[mid] if not isinstance(ref, dict) else ref.get(mid)
                except Exception:
                    name = None
            if name is None and hasattr(matlib, "name_for_id") and callable(getattr(matlib, "name_for_id")):
                try:
                    name = matlib.name_for_id(mid)
                except Exception:
                    name = None
            if isinstance(name, str):
                if hasattr(matlib, "props_for_name") and callable(getattr(matlib, "props_for_name")):
                    try:
                        E, nu = matlib.props_for_name(name)
                        return float(E), float(nu)
                    except Exception:
                        pass
                for attr in ("materials", "props", "name2props"):
                    if hasattr(matlib, attr):
                        store = getattr(matlib, attr)
                        if isinstance(store, dict) and name in store:
                            try:
                                E, nu = store[name]
                                return float(E), float(nu)
                            except Exception:
                                pass
            # 3) matlib 是 dict
            if isinstance(matlib, dict):
                if mid in matlib:
                    try:
                        E, nu = matlib[mid]
                        return float(E), float(nu)
                    except Exception:
                        pass
                cand = None
                if "id2name" in matlib:
                    try:
                        cand = matlib["id2name"][mid]
                    except Exception:
                        cand = None
                if cand is None:
                    for k in ("names", "materials_order", "enum"):
                        if k in matlib:
                            try:
                                cand = matlib[k][mid]
                                break
                            except Exception:
                                cand = None
                if isinstance(cand, str) and cand in matlib:
                    try:
                        E, nu = matlib[cand]
                        return float(E), float(nu)
                    except Exception:
                        pass
            return None

        missing: list[int] = []
        for mid in uniq:
            pair = _get_by_id(int(mid))
            if pair is None:
                missing.append(int(mid))
            else:
                lam, mu = self._E_nu_to_lame(pair[0], pair[1])
                lam_tab[int(mid)] = lam
                mu_tab[int(mid)] = mu

        if missing:
            raise KeyError(
                "[ElasticityEnergy] 无法从材料库推断 id→(E,ν)，缺失 id: " + ", ".join(map(str, missing))
            )

        return lam_tab.astype(np.float32), mu_tab.astype(np.float32)
