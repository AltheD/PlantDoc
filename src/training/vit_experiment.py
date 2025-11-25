"""
Vision Transformer / DeiT experiments (Guide.md §7.2).

Requirements:
- pip install timm

Example (ViT-B/16 with basic augmentation):

python -m src.training.vit_experiment \
    --model vit_base_patch16_224 \
    --experiment-name E4_vit_b16_basic \
    --augment basic \
    --batch-size 16 \
    --freeze-epochs 0 \
    --finetune-epochs 20 \
    --log-csv outputs/logs/vit_b16_basic.csv \
    --ckpt-dir outputs/checkpoints \
    --pred-csv outputs/predictions/vit_b16_basic_test.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Transformer baseline for PlantDoc.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/plantdoc_split_seed42.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/plantdoc_224"))
    parser.add_argument("--experiment-name", type=str, default="vit_baseline")
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--augment", choices=["none", "basic"], default="basic")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--log-csv", type=Path, default=Path("outputs/logs/vit_train_log.csv"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--pred-csv", type=Path, default=Path("outputs/predictions/vit_test_preds.csv"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(split_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    payload = json.loads(split_path.read_text())
    return payload["classes"], payload["splits"]


class PlantDocDataset(Dataset):
    def __init__(
        self,
        split_name: str,
        entries: List[str],
        class_to_idx: Dict[str, int],
        processed_root: Path,
        transform,
    ) -> None:
        self.split_name = split_name
        self.processed_root = processed_root
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.items: List[Tuple[str, str, str]] = []
        for entry in entries:
            rel = Path(entry)
            self.items.append((rel.parent.name, rel.name, entry))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        class_name, filename, raw_rel = self.items[idx]
        path = self.processed_root / self.split_name / class_name / filename
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label, raw_rel


def build_transforms(data_cfg: Dict, augment: str):
    eval_transform = create_transform(
        **data_cfg,
        is_training=False,
    )
    if augment == "none":
        return eval_transform, eval_transform

    train_transform = create_transform(
        **data_cfg,
        is_training=True,
        hflip=0.5,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.0,
    )
    return train_transform, eval_transform


def build_optimizer(params: Iterable, args: argparse.Namespace):
    if args.optimizer == "adamw":
        return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


def run_epoch(model, dataloader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    preds_all: List[int] = []
    labels_all: List[int] = []
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds_all.append(outputs.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())
    preds_tensor = torch.cat(preds_all)
    labels_tensor = torch.cat(labels_all)
    acc = (preds_tensor == labels_tensor).float().mean().item()
    macro_f1 = f1_score(labels_tensor.numpy(), preds_tensor.numpy(), average="macro")
    return {"loss": total_loss / total, "acc": acc, "macro_f1": macro_f1}


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_log(rows: List[Dict[str, float]], path: Path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(model, dataloader, classes: List[str], device: str, output_csv: Path):
    import csv

    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, raw_paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for pred, label, raw_rel in zip(preds, labels.tolist(), raw_paths):
                rows.append(
                    {
                        "image_path": raw_rel,
                        "pred_label": classes[pred],
                        "target_label": classes[label],
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pred_label", "target_label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    classes, splits = load_splits(args.split_json)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=len(classes))
    data_cfg = resolve_data_config({}, model=model)
    train_transform, eval_transform = build_transforms(data_cfg, args.augment)

    train_dataset = PlantDocDataset("train", splits["train"], class_to_idx, args.processed_root, train_transform)
    val_dataset = PlantDocDataset("val", splits["val"], class_to_idx, args.processed_root, eval_transform)
    test_dataset = PlantDocDataset("test", splits["test"], class_to_idx, args.processed_root, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), args)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.finetune_epochs))

    log_rows: List[Dict[str, float]] = []
    best_metric = -float("inf")
    best_ckpt = args.ckpt_dir / f"{args.experiment_name}_best.pt"

    # Optional freezing (useful for DeiT Tiny, etc.)
    if args.freeze_epochs > 0:
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

        head_optimizer = build_optimizer(model.get_classifier().parameters(), args)
        for epoch in range(1, args.freeze_epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, head_optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "phase": "frozen",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": head_optimizer.param_groups[0]["lr"],
            }
            log_rows.append(row)
            if val_metrics["macro_f1"] > best_metric:
                best_metric = val_metrics["macro_f1"]
                save_checkpoint(model, best_ckpt)
        for param in model.parameters():
            param.requires_grad = True

    patience = 4
    patience_counter = 0
    for epoch in range(1, args.finetune_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        log_rows.append(row)

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            patience_counter = 0
            save_checkpoint(model, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    save_log(log_rows, args.log_csv)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Best checkpoint evaluated on test: {test_metrics}")
    save_predictions(model, test_loader, classes, device, args.pred_csv)


if __name__ == "__main__":
    main()

