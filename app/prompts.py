"""提示词管理：prompt 不再写死在代码里，放在 prompts/ 目录下按版本管理。
文件：prompts/<名字>/<版本>.txt（比如 prompts/answerer/v1.txt）。
改 prompt 就改文件，不用动 agent 代码；改完重启服务生效。
加载失败（文件缺了）就直接报错——prompt 是问答的核心，缺了不能悄悄
用个藏起来的副本顶上去，那样改文件就没意义了。"""
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str, version: str = "v1") -> str:
    """读 prompts/<name>/<version>.txt 的原文（一个字不动）。
    读过一次就缓存，改了文件要重启服务才生效。"""
    path = PROMPTS_DIR / name / f"{version}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"提示词文件缺失：{path}（改 prompt 请编辑这个文件，不要改代码）"
        ) from None
