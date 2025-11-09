#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_normal_alm.py
---------------------
Augmented Lagrangian (ALM) normal-contact energy (frictionless, normal gap only).

This operator:
- Accepts per-step contact samples from ContactMap (xs, xm, n, w_area)
- Computes kinematic gap: g = ((xs + u(xs)) - (xm + u(xm))) · n
- Uses smooth negative-part: phi(g) = softplus(-g; beta)
- ALM energy: En = sum w * [ lambda * phi(g) + 0.5 * mu_n * phi(g)^2 ]
- Provides an outer update of Lagrange multipliers: lambda <- max(0, lambda + mu_n * phi(g))
- Supports per-sample weights (Weighted PINN) through w_area (and optional extra weights)
- Auto-orients normals at build-time if zero-displacement gap is predominantly negative

Intended to be called from TF2 custom training loop:
    En, stats = op.energy(u_fn, params)
    # backprop on En
    if step % K == 0:
        op.update_multipliers(u_fn, params)

Notes:
- Multipliers are per-batch; if you resample contact points each step, they are reset.
  (You can persist across steps by not calling reset_for_new_batch.)
- All internal tensors live on TF (float32 by default). NumPy inputs are converted.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------
# Utilities
# -----------------------------

def _to_tf(x, dtype=tf.float32):
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


def softplus_neg(x: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """
    Smooth negative-part: softplus(-x; beta) = (1/beta)*log(1 + exp(-beta*x))
    beta can be a scalar tensor; higher beta -> sharper.
    """
    return tf.nn.softplus(-x * beta) / (beta + 1e-12)


@dataclass
class NormalALMConfig:
    """Hyperparameters for the normal-contact ALM operator."""
    beta: float = 50.0          # softplus steepness
    mu_n: float = 1.0e3         # ALM coefficient
    enforce_nonneg_lambda: bool = True
    dtype: str = "float32"


class NormalContactALM:
    """
    Normal-contact ALM energy operator.

    Build workflow (per batch):
        op = NormalContactALM(config)
        op.build_from_numpy(xs, xm, n, w_area)       # or op.build_from_cat(cat_dict)
        En, stats = op.energy(u_fn, params)          # differentiable
        op.update_multipliers(u_fn, params)          # outer loop, not differentiable

    Where:
        - u_fn: callable (X, params)-> displ, returns TF tensor of shape (N,3)
        - params: dict carrying preload etc., if u_fn needs them; can be None
    """

    def __init__(self, cfg: Optional[NormalALMConfig] = None):
        self.cfg = cfg or NormalALMConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # Per-batch tensors (TF)
        self.xs: Optional[tf.Tensor] = None   # (N,3)
        self.xm: Optional[tf.Tensor] = None   # (N,3)
        self.n:  Optional[tf.Tensor] = None   # (N,3) (master normals; may be auto-flipped)
        self.w:  Optional[tf.Tensor] = None   # (N,)

        # State (multipliers)
        self.lmbda: Optional[tf.Variable] = None  # (N,)
        self._built_N: int = 0

        # Schedules (scalars on TF)
        self.beta = tf.Variable(self.cfg.beta, dtype=self.dtype, trainable=False, name="beta_n")
        self.mu_n = tf.Variable(self.cfg.mu_n, dtype=self.dtype, trainable=False, name="mu_n")

        # Stats cache
        self._last_gap: Optional[tf.Tensor] = None
        self._auto_flip_done: bool = False

    # ---------- building ----------

    def build_from_numpy(self, xs: np.ndarray, xm: np.ndarray, n: np.ndarray, w_area: np.ndarray,
                         extra_weights: Optional[np.ndarray] = None, auto_orient: bool = True):
        """
        Initialize per-batch tensors from NumPy arrays.
        - xs, xm: (N,3) coordinates (mm)
        - n: (N,3) normals from master surface (unit)
        - w_area: (N,) area weights from slave-surface sampling
        - extra_weights: (N,) optional additional weights (e.g., IRLS); multiplied onto w_area

        Note: This resets the multipliers to zero for the new batch.
        """
        assert xs.shape == xm.shape and xs.shape[1] == 3
        assert n.shape == xs.shape and w_area.shape[0] == xs.shape[0]

        Xs = _to_tf(xs, self.dtype)
        Xm = _to_tf(xm, self.dtype)
        Nn = _to_tf(n,  self.dtype)
        W  = _to_tf(w_area, self.dtype)

        if extra_weights is not None:
            W = W * _to_tf(extra_weights, self.dtype)

        # Normalize normals to unit (defensive)
        Nn = Nn / (tf.norm(Nn, axis=1, keepdims=True) + tf.cast(1e-12, self.dtype))

        # Assign
        self.xs, self.xm, self.n, self.w = Xs, Xm, Nn, W
        self._built_N = int(Xs.shape[0])

        # (Re)init multipliers (per-batch state)
        self.lmbda = tf.Variable(tf.zeros((self._built_N,), dtype=self.dtype), trainable=False, name="lambda_n")

        # Auto-orient normals (once per batch): ensure median((xs-xm)·n) >= 0 at zero displacement
        if auto_orient:
            self._auto_orient_normals()

        self._last_gap = None

    def build_from_cat(self, cat: Dict[str, np.ndarray], extra_weights: Optional[np.ndarray] = None, auto_orient: bool = True):
        """
        Build from concatenated dictionary returned by ContactMap.concatenate():
            cat['xs'], cat['xm'], cat['n'], cat['w_area']  (N,*)

        Optionally multiply extra per-sample weights (e.g., Weighted PINN IRLS).
        """
        self.build_from_numpy(cat["xs"], cat["xm"], cat["n"], cat["w_area"],
                              extra_weights=extra_weights, auto_orient=auto_orient)

    def _auto_orient_normals(self):
        """Flip all normals if median zero-displacement gap is negative."""
        # g0 = (xs - xm) · n
        g0 = tf.reduce_sum((self.xs - self.xm) * self.n, axis=1)
        med = tfp_median(g0)
        flip = med < tf.cast(0.0, self.dtype)
        if bool(flip.numpy()):
            self.n.assign(-self.n)

        self._auto_flip_done = True

    # ---------- core computations ----------

    def _gap(self, u_fn, params=None) -> tf.Tensor:
        """
        Compute signed gap:
            g = ((xs + u(xs)) - (xm + u(xm))) · n
        Returns:
            g: (N,) tensor
        """
        xs = tf.cast(self.xs, self.dtype)
        xm = tf.cast(self.xm, self.dtype)
        n = tf.cast(self.n, self.dtype)

        u_s = tf.cast(_ensure_2d(u_fn(xs, params)), self.dtype)
        u_m = tf.cast(_ensure_2d(u_fn(xm, params)), self.dtype)

        g = tf.reduce_sum(((xs + u_s) - (xm + u_m)) * n, axis=1)
        self._last_gap = g
        return g

    def energy(self, u_fn, params=None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute ALM normal-contact energy (scalar) and stats (dict):
            En = sum w * [ lambda * phi(g) + 0.5 * mu_n * phi(g)^2 ]
        """
        g = self._gap(u_fn, params)
        phi = softplus_neg(g, self.beta)
        En = tf.reduce_sum(self.w * (self.lmbda * phi + 0.5 * self.mu_n * phi * phi))

        # Stats for logging
        stats = {
            "min_gap": tf.reduce_min(g),
            "mean_gap": tf.reduce_mean(g),
            "pen_ratio": tf.reduce_mean(tf.cast(phi > 0.0, self.dtype)),  # fraction with penetration (phi>0)
            "phi_mean": tf.reduce_mean(phi),
        }
        return En, stats

    @tf.function(jit_compile=False)
    def update_multipliers(self, u_fn, params=None):
        """
        Outer-loop ALM update (not part of gradient path):
            lambda <- max(0, lambda + mu_n * phi(g))
        """
        g = self._gap(u_fn, params)
        phi = softplus_neg(g, self.beta)
        new_lmbda = self.lmbda + self.mu_n * phi
        if self.cfg.enforce_nonneg_lambda:
            new_lmbda = tf.maximum(new_lmbda, tf.cast(0.0, self.dtype))
        self.lmbda.assign(new_lmbda)

    # ---------- schedules / setters ----------

    def set_beta(self, beta: float):
        self.beta.assign(tf.cast(beta, self.dtype))

    def set_mu_n(self, mu_n: float):
        self.mu_n.assign(tf.cast(mu_n, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """
        Multiply current per-sample weights by extra_w (e.g., IRLS / edge weighting).
        """
        ew = _to_tf(extra_w, self.dtype)
        self.w.assign(self.w * ew)

    def reset_for_new_batch(self):
        """Clear internal tensors/state so the operator can be rebuilt with a new batch."""
        self.xs = self.xm = self.n = self.w = None
        self.lmbda = None
        self._built_N = 0
        self._last_gap = None
        self._auto_flip_done = False


# -----------------------------
# Helpers
# -----------------------------

def tfp_median(x: tf.Tensor) -> tf.Tensor:
    """
    Quick median without depending on tfp: sort and take middle.
    Assumes 1-D x.
    """
    x_sorted = tf.sort(x)
    n = tf.shape(x_sorted)[0]
    mid = n // 2
    # If even, average two middles
    even = (n % 2) == 0
    mid_val = tf.cond(
        even,
        lambda: 0.5 * (x_sorted[mid - 1] + x_sorted[mid]),
        lambda: x_sorted[mid],
    )
    return tf.cast(mid_val, x.dtype)

def _ensure_2d(u: tf.Tensor) -> tf.Tensor:
    """Ensure u has shape (N,3)."""
    if u.shape.rank == 1:
        return tf.reshape(u, (-1, 3))
    return u