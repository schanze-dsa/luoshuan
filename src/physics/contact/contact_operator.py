#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_operator.py
-------------------
Unified contact operator that wraps:
  - NormalContactALM  (normal, frictionless)
  - FrictionContactALM (tangential, Coulomb friction)

Public API (per training batch):
    op = ContactOperator(cfg)
    op.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)
    E, parts, stats = op.energy(u_fn, params)         # differentiable
    op.update_multipliers(u_fn, params)               # outer update (every K steps)
    # schedules:
    op.set_beta(beta); op.set_mu_n(mu_n)
    op.set_mu_t(mu_t); op.set_k_t(k_t); op.set_mu_f(mu_f)

Inputs `cat_dict` is typically from ContactMap.concatenate():
    xs, xm, n, t1, t2, w_area

Weighted PINN:
    Pass extra_weights (np.ndarray, shape (N,)) which will be multiplied to area weights
    for BOTH normal and friction energies.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .contact_normal_alm import NormalContactALM, NormalALMConfig, _to_tf
from .contact_friction_alm import FrictionContactALM, FrictionALMConfig


# -----------------------------
# Config for the unified operator
# -----------------------------

@dataclass
class ContactOperatorConfig:
    normal: NormalALMConfig = NormalALMConfig(beta=50.0, mu_n=1.0e3, dtype="float32")
    friction: FrictionALMConfig = FrictionALMConfig(mu_f=0.15, k_t=5.0e2, mu_t=1.0e3, dtype="float32")
    update_every_steps: int = 150   # outer ALM update cadence
    dtype: str = "float32"


class ContactOperator:
    """
    Combine normal-ALM and friction-ALM into a single, convenient interface.
    """

    def __init__(self, cfg: Optional[ContactOperatorConfig] = None):
        self.cfg = cfg or ContactOperatorConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-operators
        self.normal = NormalContactALM(self.cfg.normal)
        self.friction = FrictionContactALM(self.cfg.friction)
        self.friction.link_normal(self.normal)

        # bookkeeping
        self._built = False
        self._N = 0
        self._step = 0

    # ---------- build per batch ----------

    def build_from_cat(self, cat: Dict[str, np.ndarray],
                       extra_weights: Optional[np.ndarray] = None,
                       auto_orient: bool = True):
        """
        Build both normal and friction operators from concatenated contact arrays.
        cat must contain: xs, xm, n, t1, t2, w_area
        """
        required = ["xs", "xm", "n", "t1", "t2", "w_area"]
        for k in required:
            if k not in cat:
                raise KeyError(f"[ContactOperator] cat missing key '{k}'")

        # normal
        self.normal.build_from_cat(
            {"xs": cat["xs"], "xm": cat["xm"], "n": cat["n"], "w_area": cat["w_area"]},
            extra_weights=extra_weights, auto_orient=auto_orient
        )
        # friction (linked to normal)
        self.friction.build_from_cat(
            {"xs": cat["xs"], "xm": cat["xm"], "t1": cat["t1"], "t2": cat["t2"], "w_area": cat["w_area"]},
            extra_weights=extra_weights
        )

        self._N = int(cat["xs"].shape[0])
        self._built = True
        self._step = 0

    def reset_for_new_batch(self):
        """Clear internal state so you can rebuild with a new set of contact samples."""
        self.normal.reset_for_new_batch()
        self.friction.reset_for_new_batch()
        self._built = False
        self._N = 0
        self._step = 0

    # ---------- energy & update ----------

    def energy(self, u_fn, params=None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Compute total contact energy and return:
            E_contact_total, part_dict, stats_dict

        part_dict contains:
            {"E_n": En, "E_t": Et}

        stats_dict contains normal/friction stats merged with prefixes:
            {"n_min_gap": ..., "n_pen_ratio": ..., "t_stick_ratio": ..., ...}
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before energy().")

        En, nstats = self.normal.energy(u_fn, params)
        Et, tstats = self.friction.energy(u_fn, params)

        E = En + Et
        parts = {"E_n": En, "E_t": Et}
        stats = {
            "n_min_gap": nstats["min_gap"],
            "n_mean_gap": nstats["mean_gap"],
            "n_pen_ratio": nstats["pen_ratio"],
            "n_phi_mean": nstats["phi_mean"],
            "t_stick_ratio": tstats["stick_ratio"],
            "t_slip_ratio": tstats["slip_ratio"],
            "t_tau_trial_mean": tstats["tau_trial_mean"],
            "t_tau_mean": tstats["tau_mean"],
        }
        return E, parts, stats

    def update_multipliers(self, u_fn, params=None):
        """Outer-loop ALM update for both normal and friction."""
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before update_multipliers().")
        self.normal.update_multipliers(u_fn, params)
        self.friction.update_multipliers(u_fn, params)
        self._step += 1

    # ---------- schedules / setters ----------

    def set_beta(self, beta: float):
        self.normal.set_beta(beta)

    def set_mu_n(self, mu_n: float):
        self.normal.set_mu_n(mu_n)

    def set_mu_t(self, mu_t: float):
        self.friction.set_mu_t(mu_t)

    def set_k_t(self, k_t: float):
        self.friction.set_k_t(k_t)

    def set_mu_f(self, mu_f: float):
        # small guard: some earlier snippet had a typo; ensure correct setter here
        self.friction.mu_f.assign(tf.cast(mu_f, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """Multiply extra weights to both normal and friction energies (Weighted PINN hook)."""
        self.normal.multiply_weights(extra_w)
        self.friction.multiply_weights(extra_w)

    # ---------- convenience ----------

    @property
    def N(self) -> int:
        return self._N

    @property
    def built(self) -> bool:
        return self._built


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    # This block only checks API wiring; it does not run real contact because we lack u_fn here.
    import numpy as np
    N = 100
    cat = {
        "xs": np.random.randn(N, 3),
        "xm": np.random.randn(N, 3),
        "n":  np.tile(np.array([0., 0., 1.]), (N, 1)),
        "t1": np.tile(np.array([1., 0., 0.]), (N, 1)),
        "t2": np.tile(np.array([0., 1., 0.]), (N, 1)),
        "w_area": np.ones((N,), dtype=np.float64),
    }

    op = ContactOperator()
    op.build_from_cat(cat, extra_weights=None, auto_orient=True)

    # dummy u_fn: zero displacement
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.zeros_like(X)

    E, parts, stats = op.energy(u_fn)
    print("E_contact =", float(E.numpy()))
    print("parts:", {k: float(v.numpy()) for k, v in parts.items()})
    print("stats keys:", list(stats.keys()))
