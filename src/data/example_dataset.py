from pathlib import Path
from typing import List, Tuple


def list_images_under(root: Path) -> List[Path]:
    """
    简单示例：列出目录下的所有 jpg/jpeg/png 文件（递归）。
    """
    patterns = ("*.jpg", "*.jpeg", "*.png")
    results: List[Path] = []
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    return results


def demo() -> Tuple[int, List[Path]]:
    """
    返回图片数量与前若干个样例路径，示例用途。
    """
    dataset_root = Path("data/raw/PlantDoc")
    files = list_images_under(dataset_root)
    preview = files[:5]
    return len(files), preview


if __name__ == "__main__":
    total, preview = demo()
    print(f"共找到图像文件: {total}")
    for p in preview:
        print(f"- {p}")


