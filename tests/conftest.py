import sys
from pathlib import Path


# 测试直接使用当前 enhance 工作区内的 StarVLA 兼容层，避免误导入其他 editable checkout。
REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_RUNTIME = REPO_ROOT / "third_party" / "starvla_runtime"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STARVLA_RUNTIME))
