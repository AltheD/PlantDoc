from typing import Any
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """
    一个极简 CNN，用于示例与快速连通性检查。
    输入：3x224x224
    输出：num_classes logits
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def demo_forward(batch_size: int = 2, num_classes: int = 3) -> Any:
    model = TinyCNN(num_classes=num_classes)
    dummy = torch.randn(batch_size, 3, 224, 224)
    logits = model(dummy)
    return logits.shape


if __name__ == "__main__":
    shape = demo_forward()
    print(f"logits shape: {shape}")


