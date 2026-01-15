#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a TensorFlow SavedModel from an existing training checkpoint directory.

This is useful when training finished but SavedModel export failed, or when you
want to export without running another long training session.

Example:
  python tools/export_from_ckpt.py --ckpt checkpoints/run-20251215-210500 --export results/saved_model_from_ckpt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


def _default_export_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path("results") / f"saved_model_from_ckpt_{ts}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (used to rebuild the model architecture).",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint run directory, e.g. checkpoints/run-YYYYMMDD-HHMMSS",
    )
    ap.add_argument(
        "--export",
        default="",
        help="Output SavedModel directory (default: results/saved_model_from_ckpt_TIMESTAMP).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    ckpt_dir = Path(str(args.ckpt)).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    export_dir = str(Path(args.export).expanduser().resolve()) if args.export else _default_export_dir()

    import tensorflow as tf

    import main as main_mod
    from train.trainer import Trainer

    cfg, asm = main_mod._prepare_config_with_autoguess()
    cfg.ckpt_dir = str(ckpt_dir)

    # For exporting weights, mixed precision is unnecessary and can be a source
    # of numerical quirks; keep it off unless you explicitly need it.
    cfg.mixed_precision = None

    trainer = Trainer(cfg)
    trainer.build()

    latest = None
    if getattr(trainer, "ckpt_manager", None) is not None:
        latest = trainer.ckpt_manager.latest_checkpoint
    if not latest:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")

    # Restore only model weights (encoder/field) to avoid optimizer mismatch.
    ckpt = tf.train.Checkpoint(encoder=trainer.model.encoder, field=trainer.model.field)
    ckpt.restore(latest).expect_partial()
    print(f"[export] restored -> {latest}")

    trainer.export_saved_model(export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

