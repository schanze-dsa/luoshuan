#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loss_energy.py
--------------
Total potential energy assembly for DFEM/PINN with contact & preloads.

Composition:
    Π = w_int * E_int
        + w_cn * E_n
        + w_ct * E_t
        + w_tie * E_tie
        + w_bc  * E_bc
        - w_pre * W_pre

Public usage (typical):
    # 1) Build sub-operators per batch
    elas.build_from_numpy(X_vol, w_vol, mat_id, mat_lib)
    contact.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)
    tie.build_from_numpy(xs, xm, w_area, dof_mask=None)
    bc.build_from_numpy(X_bc, dof_mask, u_target, w_bc)

    # 2) Assemble total energy
    total = TotalEnergy()
    total.attach(elasticity=elas, contact=contact, preload=preload, ties=[tie], bcs=[bc])

    # 3) Compute energy & update multipliers in training loop
    E, parts, stats = total.energy(model.u_fn, params={"P": [P1,P2,P3]})
    if step % total.cfg.update_every_steps == 0:
        total.update_multipliers(model.u_fn, params)

Weighted PINN:
    - You can multiply extra per-sample weights into components:
        contact.multiply_weights(w_contact)
        for t in ties: t.multiply_weights(w_tie)
        for b in bcs:  b.multiply_weights(w_bc)
    - If you need to reweight volume points, see TotalEnergy.scale_volume_weights().

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

# sub-operators
from physics.elasticity_energy import ElasticityEnergy, ElasticityConfig
from physics.contact.contact_operator import ContactOperator, ContactOperatorConfig
from physics.tie_constraints import TiePenalty, TieConfig
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig
from physics.preload_model import PreloadWork, PreloadConfig


# -----------------------------
# Config for total energy
# -----------------------------

@dataclass
class TotalConfig:
    # coefficients for each term
    w_int: float = 1.0
    w_cn: float = 1.0            # normal contact
    w_ct: float = 1.0            # frictional contact
    w_tie: float = 1.0
    w_bc: float = 1.0
    w_pre: float = 1.0           # multiplies the subtracted W_pre (so larger -> stronger preload driving)

    # ALM outer update cadence for contact (can be used by training loop)
    update_every_steps: int = 150

    # dtype
    dtype: str = "float32"


# -----------------------------
# Total energy assembler
# -----------------------------

class TotalEnergy:
    """
    Assemble total potential energy from provided operators.
    """

    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-ops (optional ones can be None)
        self.elasticity: Optional[ElasticityEnergy] = None
        self.contact: Optional[ContactOperator] = None
        self.ties: List[TiePenalty] = []
        self.bcs: List[BoundaryPenalty] = []
        self.preload: Optional[PreloadWork] = None

        # trainable (non) scalars as TF vars so they can be scheduled
        self.w_int = tf.Variable(self.cfg.w_int, dtype=self.dtype, trainable=False, name="w_int")
        self.w_cn  = tf.Variable(self.cfg.w_cn,  dtype=self.dtype, trainable=False, name="w_cn")
        self.w_ct  = tf.Variable(self.cfg.w_ct,  dtype=self.dtype, trainable=False, name="w_ct")
        self.w_tie = tf.Variable(self.cfg.w_tie, dtype=self.dtype, trainable=False, name="w_tie")
        self.w_bc  = tf.Variable(self.cfg.w_bc,  dtype=self.dtype, trainable=False, name="w_bc")
        self.w_pre = tf.Variable(self.cfg.w_pre, dtype=self.dtype, trainable=False, name="w_pre")

        self._built = False

    # ---------- wiring ----------

    def attach(self,
               elasticity: Optional[ElasticityEnergy] = None,
               contact: Optional[ContactOperator] = None,
               preload: Optional[PreloadWork] = None,
               ties: Optional[List[TiePenalty]] = None,
               bcs: Optional[List[BoundaryPenalty]] = None):
        """
        Attach sub-components built for the current batch.
        """
        if elasticity is not None:
            self.elasticity = elasticity
        if contact is not None:
            self.contact = contact
        if preload is not None:
            self.preload = preload
        if ties is not None:
            self.ties = list(ties)
        if bcs is not None:
            self.bcs = list(bcs)

        self._built = True

    def reset(self):
        """Detach everything (e.g., before building a new batch)."""
        self.elasticity = None
        self.contact = None
        self.preload = None
        self.ties = []
        self.bcs = []
        self._built = False

    # ---------- optional helpers ----------

    def scale_volume_weights(self, factor: float):
        """
        Multiply all volume quadrature weights by 'factor' (coarse reweighting).
        Use this if you want Weighted PINN-like emphasis on volume PDE residuals.
        """
        if self.elasticity is None or self.elasticity.w is None:
            return
        self.elasticity.w.assign(self.elasticity.w * tf.cast(factor, self.dtype))

    # ---------- energy ----------

    def energy(self, u_fn, params=None, tape=None):
        """
        Compute total potential and return:
            Π_total, parts_dict, stats_dict

        parts_dict:
            {'E_int':..., 'E_n':..., 'E_t':..., 'E_tie':..., 'E_bc':..., 'W_pre':...}

        stats_dict: merged sub-stats with prefixes
        """
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")

        parts: Dict[str, tf.Tensor] = {}
        stats: Dict[str, tf.Tensor] = {}

        # Elasticity
        E_int = tf.cast(0.0, self.dtype)
        if self.elasticity is not None:
            # 关键：把 tape 传进弹性项
            E_int, estates = self.elasticity.energy(u_fn, params, tape=tape)
            parts["E_int"] = E_int
            stats.update({f"el_{k}": v for k, v in estates.items()})

        # Contact
        E_n = tf.cast(0.0, self.dtype)
        E_t = tf.cast(0.0, self.dtype)
        if self.contact is not None:
            E_c, cparts, cstats = self.contact.energy(u_fn, params)
            E_n = cparts["E_n"]
            E_t = cparts["E_t"]
            parts["E_n"] = E_n
            parts["E_t"] = E_t
            stats.update(cstats)

        # Tie (sum over multiple)
        E_tie = tf.cast(0.0, self.dtype)
        if self.ties:
            tie_acc = []
            for i, t in enumerate(self.ties):
                Ei, si = t.energy(u_fn, params)
                tie_acc.append(Ei)
                stats.update({f"tie{i+1}_{k}": v for k, v in si.items()})
            E_tie = tf.add_n(tie_acc)
            parts["E_tie"] = E_tie

        # Boundary (sum over multiple)
        E_bc = tf.cast(0.0, self.dtype)
        if self.bcs:
            bc_acc = []
            for i, b in enumerate(self.bcs):
                Ei, si = b.energy(u_fn, params)
                bc_acc.append(Ei)
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            E_bc = tf.add_n(bc_acc)
            parts["E_bc"] = E_bc

        # Preload work (positive value to be subtracted)
        W_pre = tf.cast(0.0, self.dtype)
        if self.preload is not None:
            W_pre, pstats = self.preload.energy(u_fn, params)
            parts["W_pre"] = W_pre
            stats.update({f"pre_{k}": v for k, v in pstats.items()})

        # Assemble Π
        Pi = ( self.w_int * E_int
             + self.w_cn  * E_n
             + self.w_ct  * E_t
             + self.w_tie * E_tie
             + self.w_bc  * E_bc
             - self.w_pre * W_pre )

        return Pi, parts, stats

    # ---------- outer updates ----------

    def update_multipliers(self, u_fn, params=None):
        """
        Run ALM outer-loop updates for contact (and anything else in future).
        Call this every cfg.update_every_steps steps in your training loop.
        """
        if self.contact is not None:
            self.contact.update_multipliers(u_fn, params)

    # ---------- setters / schedules ----------

    def set_coeffs(self,
                   w_int: Optional[float] = None,
                   w_cn: Optional[float] = None,
                   w_ct: Optional[float] = None,
                   w_tie: Optional[float] = None,
                   w_bc: Optional[float] = None,
                   w_pre: Optional[float] = None):
        """Set any subset of coefficients on the fly (e.g., curriculum)."""
        if w_int is not None: self.w_int.assign(tf.cast(w_int, self.dtype))
        if w_cn  is not None: self.w_cn.assign(tf.cast(w_cn,  self.dtype))
        if w_ct  is not None: self.w_ct.assign(tf.cast(w_ct,  self.dtype))
        if w_tie is not None: self.w_tie.assign(tf.cast(w_tie, self.dtype))
        if w_bc  is not None: self.w_bc.assign(tf.cast(w_bc,  self.dtype))
        if w_pre is not None: self.w_pre.assign(tf.cast(w_pre, self.dtype))


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    # Dummy operators with toy data to verify API wiring
    import numpy as np
    from physics.material_lib import MaterialLibrary

    # 1) Elasticity
    matlib = MaterialLibrary({"steel": (210000.0, 0.3)})
    Nvol = 256
    X_vol = np.random.randn(Nvol, 3).astype(np.float64)
    w_vol = np.ones((Nvol,), dtype=np.float64)
    mat_id = np.zeros((Nvol,), dtype=np.int64)
    elas = ElasticityEnergy(ElasticityConfig())
    elas.build_from_numpy(X_vol, w_vol, mat_id, matlib)

    # 2) Contact (random placeholders)
    from physics.contact.contact_operator import ContactOperator
    cat = {
        "xs": np.random.randn(128, 3),
        "xm": np.random.randn(128, 3),
        "n":  np.tile(np.array([0., 0., 1.]), (128, 1)),
        "t1": np.tile(np.array([1., 0., 0.]), (128, 1)),
        "t2": np.tile(np.array([0., 1., 0.]), (128, 1)),
        "w_area": np.ones((128,), dtype=np.float64),
    }
    contact = ContactOperator()
    contact.build_from_cat(cat)

    # 3) Tie + BC (toy)
    tie = TiePenalty(TieConfig())
    tie.build_from_numpy(xs=cat["xs"], xm=cat["xm"], w_area=cat["w_area"])
    bc = BoundaryPenalty(BoundaryConfig())
    X_bc = np.random.randn(64, 3)
    mask = np.ones((64, 3))
    bc.build_from_numpy(X_bc, mask, u_target=None, w_bc=None)

    # 4) Preload (toy)
    from physics.preload_model import PreloadWork
    pl = PreloadWork()
    n_axis = np.array([0, 0, 1.0])
    bolts = [("bolt1", n_axis, np.random.randn(64,3), np.ones(64), np.random.randn(64,3), np.ones(64)),
             ("bolt2", n_axis, np.random.randn(64,3), np.ones(64), np.random.randn(64,3), np.ones(64)),
             ("bolt3", n_axis, np.random.randn(64,3), np.ones(64), np.random.randn(64,3), np.ones(64))]
    pl.build_from_numpy(bolts)

    # 5) Model forward (dummy)
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        # small quadratic displacement in z
        uz = 1e-3 * (X[:, 0:1] ** 2 + X[:, 1:2] ** 2)
        return tf.concat([tf.zeros_like(uz), tf.zeros_like(uz), uz], axis=1)

    total = TotalEnergy(TotalConfig(update_every_steps=50))
    total.attach(elasticity=elas, contact=contact, preload=pl, ties=[tie], bcs=[bc])

    P = tf.constant([300.0, 500.0, 700.0], dtype=tf.float32)
    Pi, parts, stats = total.energy(u_fn, params={"P": P})
    print("Π =", float(Pi.numpy()))
    print({k: float(v.numpy()) for k, v in parts.items()})
    total.update_multipliers(u_fn, params={"P": P})
