"""脚本启动时，题库目录不能参与第三方包的查找。"""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["run_eval.py", "run_eval_auto.py"])
def test_eval_data_is_not_on_import_path(script):
    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    # 重现 python eval/<script> 的搜索路径，只加载入口，避免运行整轮评测。
    probe = """
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1])
sys.path[0] = str(script.parent)
runpy.run_path(str(script))
assert script.parent not in [Path(p).resolve() for p in sys.path]
"""
    subprocess.run([sys.executable, "-c", probe, str(eval_dir / script)], check=True)
