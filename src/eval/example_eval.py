from typing import Dict
import torch

from src.models.example_model import TinyCNN


def evaluate_dummy() -> Dict[str, float]:
    """
    最小评估示例：随机输入计算 Top-1 准确率（随机标签），用于连通性检查。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN(num_classes=3).to(device)
    model.eval()

    inputs = torch.randn(8, 3, 224, 224, device=device)
    targets = torch.randint(0, 3, (8,), device=device)
    with torch.no_grad():
        logits = model(inputs)
        acc = (logits.argmax(dim=1) == targets).float().mean().item()
    return {"acc": float(acc)}


if __name__ == "__main__":
    metrics = evaluate_dummy()
    print(metrics)


