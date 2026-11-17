"""
原项目复现启动器
在导入 pytorch_lightning 之前，为 Python 3.12 打上 pkgutil 兼容补丁
（setuptools 旧版本依赖已移除的 ImpImporter/ImpLoader），
然后原样运行 flow-matching-mnist-main/train.py。
"""
import pkgutil
import sys
import os

# ---- Python 3.12 兼容补丁 ----
if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        def __init__(self, path=None):
            self.path = path
    pkgutil.ImpImporter = ImpImporter

if not hasattr(pkgutil, 'ImpLoader'):
    class ImpLoader:
        def __init__(self, fullname, path, file, filename, etc):
            self.fullname = fullname
            self.path = path
            self.file = file
            self.filename = filename
            self.etc = etc
    pkgutil.ImpLoader = ImpLoader

# 尝试用 distutils hack 兼容
try:
    import setuptools  # noqa: F401
    print("setuptools 加载 OK")
except Exception as e:
    print(f"setuptools 警告（可忽略）: {e}")

# 切换到原项目目录并运行
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flow-matching-mnist-main')
os.chdir(project_dir)
sys.path.insert(0, project_dir)

print(f"运行目录: {project_dir}")
print(f"命令: python train.py")

import runpy
runpy.run_path(os.path.join(project_dir, 'train.py'), run_name='__main__')
