# list_surfaces.py
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path: sys.path.insert(0, SRC)

from inp_io.inp_parser import load_inp  # 注意：我们把 io 重命名成 inp_io 了
INP_PATH = r"D:\shuangfan\shuangfan.inp"   # ←改成你的实际路径

asm = load_inp(INP_PATH)
print("=== Parts ===")
for k in asm.parts.keys():
    print("  ", k)
print("\n=== Surfaces (keys) ===")
for k, s in asm.surfaces.items():
    print(f'  {k}')
print("\n=== Surfaces (pretty names) ===")
for k, s in asm.surfaces.items():
    print(f'  key={k}   name={s.name}')
