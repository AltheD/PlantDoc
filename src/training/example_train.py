from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.example_model import TinyCNN


def train_one_step() -> Dict[str, float]:
    """
    最小可运行训练示例：单步前向+反向，用于验证环境。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    inputs = torch.randn(4, 3, 224, 224, device=device)
    targets = torch.randint(0, 3, (4,), device=device)

    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    acc = (logits.argmax(dim=1) == targets).float().mean().item()

    return {"loss": float(loss.item()), "acc": float(acc)}


if __name__ == "__main__":
    metrics = train_one_step()
    print(metrics)


