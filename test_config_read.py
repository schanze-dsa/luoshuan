#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试：检查TrainerConfig的max_steps是否正确读取
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train.trainer import TrainerConfig
import yaml

# 读取config
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg_yaml = yaml.safe_load(f)

optimizer_cfg = cfg_yaml.get("optimizer_config", {})
epochs_value = optimizer_cfg.get("epochs")

print(f"config.yaml中的epochs值: {epochs_value}")
print(f"TrainerConfig默认max_steps: {TrainerConfig.max_steps}")

# 模拟main.py的逻辑
train_steps = int(optimizer_cfg.get("epochs", TrainerConfig.max_steps))
print(f"计算得到的train_steps: {train_steps}")

# 创建配置
cfg = TrainerConfig(
    inp_path="shuangfan.inp",
    max_steps=train_steps
)
print(f"cfg.max_steps: {cfg.max_steps}")
print(f"cfg.adam_steps: {cfg.adam_steps}")
