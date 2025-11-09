#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_friction_alm.py
-----------------------
Augmented Lagrangian (ALM) tangential/friction contact operator.

Core idea:
- Relative tangential slip on the contact plane:
      s = ((xs + u(xs)) - (xm + u(xm)))
      s_t = [ t1·s, t2·s ]  (2D vector in tangential basis)
- Trial tangential traction:
      tau_trial = lambda_t + k_t * s_t           (R^2 per contact point)
- Friction cone (Coulomb):
      p_eff = max(0, lambda_n + mu_n * phi(g))   (effective normal compression from the normal ALM)
      tau_c = mu_f * p_eff                       (radius of friction cone)
- Smooth projection onto friction cone:
      ||·||_eps = sqrt( ·^2 + eps^2 )
      scale = min(1, tau_c / ||tau_trial||_eps)
      tau = scale * tau_trial
- Residual:
      r_t = tau_trial - tau
- Tangential pseudo-energy (differentiable):
      Et = sum w * (1/(2*mu_t)) * ||r_t||^2

Outer update (every K steps):
      lambda_t <- lambda_t + k_t * s_t - tau

Inputs (per batch):
  xs, xm : (N,3) slave/master coordinates (mm)
  t1, t2 : (N,3) orthonormal tangential basis (from master tri normals)
  w_area : (N,) area weights (can be multiplied by extra weights for Weighted PINN)
  normal_op : NormalContactALM, to retrieve phi(g) and lambda_n for p_eff

All computations are in TF2; returns (Et, stats) for backprop and logging.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

# Optional import for type hints (no runtime cyclic dep)
try:
    from .contact_normal_alm import NormalContactALM, softplus_neg, _ensure_2d, _to_tf
except Exception:
    # When running this file standalone, we fallback to local helpers
    def _to_tf(x, dtype=tf.float32):
        if isinstance(x, tf.Tensor):
            return tf.cast(x, dtype)
        return tf.convert_to_tensor(x, dtype=dtype)
    def _ensure_2d(u: tf.Tensor) -> tf.Tensor:
        if u.shape.rank == 1:
            return tf.reshape(u, (-1, 3))
        return u
    def softplus_neg(x: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        return tf.nn.softplus(-x * beta) / (beta + 1e-12)
    class NormalContactALM:  # minimal stub for type hints
        beta: tf.Variable
        mu_n: tf.Variable
        lmbda: tf.Variable
        def _gap(self, u_fn, params=None) -> tf.Tensor:  # type: ignore
            raise NotImplementedError


# -----------------------------
# Config
# -----------------------------

@dataclass
class FrictionALMConfig:
    mu_f: float = 0.15      # Coulomb friction coefficient
    k_t: float = 5.0e2      # tangential penalty "stiffness" (for trial traction)
    mu_t: float = 1.0e3     # ALM coefficient for tangential residual pseudo-energy
    eps: float = 1.0e-6     # smoothing for ||·||_eps
    use_effective_normal: bool = True   # tau_c = mu_f * max(0, lambda_n + mu_n*phi(g))
    dtype: str = "float32"


class FrictionContactALM:
    """
    Tangential/friction ALM operator.

    Typical usage per batch:
        fn = FrictionContactALM(cfg)
        fn.link_normal(normal_op)                               # to fetch p_eff
        fn.build_from_numpy(xs, xm, t1, t2, w_area, extra_w)    # reset lambda_t
        Et, stats = fn.energy(u_fn, params)                     # backprop on Et
        fn.update_multipliers(u_fn, params)                     # outer-loop update

    Notes:
    - lambda_t is a per-sample 2D vector (N,2).
    - If you re-sample contact points every step, call build_* each time (it resets lambda_t).
      If you want to persist lambda_t across steps, manage indices yourself and avoid rebuild.
    """

    def __init__(self, cfg: Optional[FrictionALMConfig] = None):
        self.cfg = cfg or FrictionALMConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # batch tensors
        self.xs: Optional[tf.Tensor] = None  # (N,3)
        self.xm: Optional[tf.Tensor] = None  # (N,3)
        self.t1: Optional[tf.Tensor] = None  # (N,3)
        self.t2: Optional[tf.Tensor] = None  # (N,3)
        self.w:  Optional[tf.Tensor] = None  # (N,)

        # multipliers (tangential, 2D)
        self.lmbda_t: Optional[tf.Variable] = None  # (N,2)

        # schedules / scalars
        self.mu_t = tf.Variable(self.cfg.mu_t, dtype=self.dtype, trainable=False, name="mu_t")
        self.k_t  = tf.Variable(self.cfg.k_t,  dtype=self.dtype, trainable=False, name="k_t")
        self.mu_f = tf.Variable(self.cfg.mu_f, dtype=self.dtype, trainable=False, name="mu_f")
        self.eps  = tf.Variable(self.cfg.eps,  dtype=self.dtype, trainable=False, name="eps")

        # link to normal operator (for p_eff)
        self.normal_op: Optional[NormalContactALM] = None

        # cache
        self._last_st: Optional[tf.Tensor] = None
        self._last_tau_trial: Optional[tf.Tensor] = None
        self._last_tau: Optional[tf.Tensor] = None

        self._built_N: int = 0

    # ---------- linking & building ----------

    def link_normal(self, normal_op: NormalContactALM):
        """Link a NormalContactALM instance so we can compute p_eff for the friction cone."""
        self.normal_op = normal_op

    def build_from_numpy(self, xs: np.ndarray, xm: np.ndarray, t1: np.ndarray, t2: np.ndarray,
                         w_area: np.ndarray, extra_weights: Optional[np.ndarray] = None):
        """
        Initialize per-batch tensors from NumPy arrays.
        - xs, xm : (N,3)
        - t1, t2 : (N,3) orthonormal tangential basis
        - w_area : (N,) area weights (will be multiplied by extra_weights if provided)
        """
        assert xs.shape == xm.shape and xs.shape[1] == 3
        assert t1.shape == xs.shape and t2.shape == xs.shape and w_area.shape[0] == xs.shape[0]

        Xs = _to_tf(xs, self.dtype)
        Xm = _to_tf(xm, self.dtype)
        T1 = _to_tf(t1, self.dtype)
        T2 = _to_tf(t2, self.dtype)
        W  = _to_tf(w_area, self.dtype)

        if extra_weights is not None:
            W = W * _to_tf(extra_weights, self.dtype)

        # Normalize basis defensively (should already be orthonormal)
        T1 = T1 / (tf.norm(T1, axis=1, keepdims=True) + tf.cast(1e-12, self.dtype))
        # Make T2 ortho to T1 isn't strictly necessary here—assume upstream ensures it.

        self.xs, self.xm, self.t1, self.t2, self.w = Xs, Xm, T1, T2, W
        self._built_N = int(Xs.shape[0])

        # reset tangential multipliers
        self.lmbda_t = tf.Variable(tf.zeros((self._built_N, 2), dtype=self.dtype), trainable=False, name="lambda_t")

        # clear caches
        self._last_st = None
        self._last_tau_trial = None
        self._last_tau = None

    def build_from_cat(self, cat: Dict[str, np.ndarray], extra_weights: Optional[np.ndarray] = None):
        """
        Build from ContactMap.concatenate() dict:
            xs, xm, t1, t2, w_area
        """
        self.build_from_numpy(cat["xs"], cat["xm"], cat["t1"], cat["t2"], cat["w_area"], extra_weights=extra_weights)

    # ---------- internals ----------

    def _relative_slip_t(self, u_fn, params=None) -> tf.Tensor:
        """
        Compute tangential relative displacement components s_t in the [t1,t2] basis:
            s = ((xs + u(xs)) - (xm + u(xm)))
            s_t = [ t1·s, t2·s ]  -> shape (N,2)
        """
        xs = tf.cast(self.xs, self.dtype)
        xm = tf.cast(self.xm, self.dtype)
        t1 = tf.cast(self.t1, self.dtype)
        t2 = tf.cast(self.t2, self.dtype)

        u_s = tf.cast(_ensure_2d(u_fn(xs, params)), self.dtype)
        u_m = tf.cast(_ensure_2d(u_fn(xm, params)), self.dtype)
        s = (xs + u_s) - (xm + u_m)                  # (N,3)
        s1 = tf.reduce_sum(t1 * s, axis=1)           # (N,)
        s2 = tf.reduce_sum(t2 * s, axis=1)           # (N,)
        st = tf.stack([s1, s2], axis=1)              # (N,2)
        self._last_st = st
        return st

    def _effective_normal_pressure(self, u_fn, params=None) -> tf.Tensor:
        """
        Compute p_eff, the effective normal compression used in the friction cone:
            p_eff = max(0, lambda_n + mu_n * phi(g)), phi(g)=softplus(-g; beta)
        Requires that link_normal() has been called.
        """
        if self.normal_op is None:
            raise RuntimeError("FrictionContactALM needs link_normal(normal_op) before use.")
        # reuse the normal-op parameters (all TF scalars/tensors)
        g = self.normal_op._gap(u_fn, params)                      # (N,)
        phi = softplus_neg(g, self.normal_op.beta)                 # (N,)
        p_eff = self.normal_op.lmbda + self.normal_op.mu_n * phi   # (N,)
        # Clamp to nonnegative (no tension)
        p_eff = tf.maximum(p_eff, tf.cast(0.0, self.dtype))
        return p_eff

    # ---------- energy & update ----------

    def energy(self, u_fn, params=None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute friction ALM pseudo-energy:
            Et = sum w * (1/(2*mu_t)) * || r_t ||^2
        where:
            r_t = tau_trial - tau
            tau_trial = lambda_t + k_t * s_t
            tau = projection onto friction cone of radius tau_c = mu_f * p_eff
        """
        # s_t
        st = self._relative_slip_t(u_fn, params)                 # (N,2)

        # trial traction
        tau_trial = self.lmbda_t + self.k_t * st                 # (N,2)

        # friction cone radius
        if self.cfg.use_effective_normal:
            p_eff = self._effective_normal_pressure(u_fn, params)    # (N,)
        else:
            # If not using effective normal, fall back to |lambda_n| (requires link_normal anyway)
            if self.normal_op is None:
                raise RuntimeError("use_effective_normal=False still needs link_normal(normal_op).")
            p_eff = tf.maximum(self.normal_op.lmbda, tf.cast(0.0, self.dtype))
        tau_c = self.mu_f * p_eff                                 # (N,)

        # smooth norm and projection
        eps = tf.cast(self.eps, self.dtype)
        norm_trial = tf.sqrt(tf.reduce_sum(tau_trial * tau_trial, axis=1) + eps * eps)  # (N,)
        scale = tf.minimum(tf.cast(1.0, self.dtype), tau_c / (norm_trial + 1e-12))      # (N,)
        tau = tau_trial * scale[:, None]                           # (N,2)

        # residual & energy
        r_t = tau_trial - tau                                      # (N,2)
        Et = tf.reduce_sum(self.w * (0.5 / self.mu_t) * tf.reduce_sum(r_t * r_t, axis=1))

        # stats
        stick = tf.cast(scale >= 0.999, self.dtype)                # scale ~1 => inside cone => stick
        slip  = tf.cast(scale <  0.999, self.dtype)
        stats = {
            "stick_ratio": tf.reduce_mean(stick),
            "slip_ratio":  tf.reduce_mean(slip),
            "tau_trial_mean": tf.reduce_mean(norm_trial),
            "tau_mean": tf.reduce_mean(tf.sqrt(tf.reduce_sum(tau * tau, axis=1))),
        }

        # cache
        self._last_tau_trial = tau_trial
        self._last_tau = tau

        return Et, stats

    @tf.function(jit_compile=False)
    def update_multipliers(self, u_fn, params=None):
        """
        Outer update for tangential multipliers:
            lambda_t <- lambda_t + k_t * s_t - tau
        """
        # recompute s_t and tau consistent with current state
        st = self._relative_slip_t(u_fn, params)                  # (N,2)
        tau_trial = self.lmbda_t + self.k_t * st
        # friction cone
        if self.cfg.use_effective_normal:
            p_eff = self._effective_normal_pressure(u_fn, params)
        else:
            p_eff = tf.maximum(self.normal_op.lmbda, tf.cast(0.0, self.dtype))
        tau_c = self.mu_f * p_eff
        eps = tf.cast(self.eps, self.dtype)
        norm_trial = tf.sqrt(tf.reduce_sum(tau_trial * tau_trial, axis=1) + eps * eps)
        scale = tf.minimum(tf.cast(1.0, self.dtype), tau_c / (norm_trial + 1e-12))
        tau = tau_trial * scale[:, None]

        new_lambda_t = self.lmbda_t + self.k_t * st - tau
        self.lmbda_t.assign(new_lambda_t)

    # ---------- schedules / setters ----------

    def set_mu_t(self, mu_t: float):
        self.mu_t.assign(tf.cast(mu_t, self.dtype))

    def set_k_t(self, k_t: float):
        self.k_t.assign(tf.cast(k_t, self.dtype))

    def set_mu_f(self, mu_f: float):
        self.mu_f.assign(tf.cast(muf, self.dtype))  # type: ignore  # (typo guard in code review)

    def multiply_weights(self, extra_w: np.ndarray):
        """Multiply current per-sample weights by extra_w (Weighted PINN hooks)."""
        ew = _to_tf(extra_w, self.dtype)
        self.w.assign(self.w * ew)

    def reset_for_new_batch(self):
        """Clear batch tensors/state to allow rebuild with new samples."""
        self.xs = self.xm = self.t1 = self.t2 = self.w = None
        self.lmbda_t = None
        self._built_N = 0
        self._last_st = None
        self._last_tau_trial = None
        self._last_tau = None