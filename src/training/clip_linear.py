"""
CLIP feature extraction + linear probe (Guide.md §7.2.E5).

Requirements:
- pip install git+https://github.com/openai/CLIP.git

Example:

python -m src.training.clip_linear \
    --experiment-name E5_clip_vitb32_linear \
    --batch-size 64 \
    --epochs 10 \
    --log-csv outputs/logs/clip_linear.csv \
    --pred-csv outputs/predictions/clip_linear_test.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP feature linear probe for PlantDoc.")
    parser.add_argument("--split-json", type=Path, default=Path("data/splits/plantdoc_split_seed42.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/plantdoc_224"))
    parser.add_argument("--experiment-name", type=str, default="clip_linear")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-csv", type=Path, default=Path("outputs/logs/clip_linear_log.csv"))
    parser.add_argument("--pred-csv", type=Path, default=Path("outputs/predictions/clip_linear_test.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(split_path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    payload = json.loads(split_path.read_text())
    return payload["classes"], payload["splits"]


class PlantDocClipDataset(Dataset):
    def __init__(self, split_name: str, entries: List[str], class_to_idx: Dict[str, int], processed_root: Path, preprocess) -> None:
        self.split_name = split_name
        self.processed_root = processed_root
        self.preprocess = preprocess
        self.class_to_idx = class_to_idx
        self.items = []
        for entry in entries:
            rel = Path(entry)
            self.items.append((rel.parent.name, rel.name, entry))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        class_name, filename, raw_rel = self.items[idx]
        path = self.processed_root / self.split_name / class_name / filename
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)
        label = self.class_to_idx[class_name]
        return image, label, raw_rel


def save_log(rows, path: Path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, paths: List[str]):
        self.features = features
        self.labels = labels
        self.paths = paths

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.paths[idx]


def extract_features(model, dataloader, device):
    all_feats = []
    all_labels = []
    all_paths = []
    model.eval()
    with torch.no_grad():
        for images, labels, raw_paths in tqdm(dataloader, desc="Extracting CLIP features"):
            images = images.to(device)
            feats = model.encode_image(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_paths.extend(raw_paths)
    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    return features, labels, all_paths


def main():
    args = parse_args()
    set_seed(args.seed)
    classes, splits = load_splits(args.split_json)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    clip_model, preprocess = clip.load(args.clip_model, device=args.device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    train_dataset = PlantDocClipDataset("train", splits["train"], class_to_idx, args.processed_root, preprocess)
    val_dataset = PlantDocClipDataset("val", splits["val"], class_to_idx, args.processed_root, preprocess)
    test_dataset = PlantDocClipDataset("test", splits["test"], class_to_idx, args.processed_root, preprocess)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_feats, train_labels, _ = extract_features(clip_model, train_loader, args.device)
    val_feats, val_labels, _ = extract_features(clip_model, val_loader, args.device)
    test_feats, test_labels, test_paths = extract_features(clip_model, test_loader, args.device)

    feature_dim = train_feats.shape[1]
    linear_head = nn.Linear(feature_dim, len(classes)).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_feat_dataset = TensorDataset(train_feats, train_labels)
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size, shuffle=True)

    log_rows = []
    best_metric = -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        linear_head.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for feats, labels_tensor in train_feat_loader:
            feats = feats.to(args.device)
            labels_tensor = labels_tensor.to(args.device)
            optimizer.zero_grad()
            outputs = linear_head(feats)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_tensor).sum().item()
            total += feats.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        linear_head.eval()
        with torch.no_grad():
            val_outputs = linear_head(val_feats.to(args.device))
            val_loss = criterion(val_outputs, val_labels.to(args.device)).item()
            val_preds = val_outputs.argmax(dim=1).cpu()
            val_acc = (val_preds == val_labels).float().mean().item()
            val_macro_f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average="macro")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        }
        log_rows.append(row)
        print(row)

        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            best_state = linear_head.state_dict()

    save_log(log_rows, args.log_csv)
    linear_head.load_state_dict(best_state)

    with torch.no_grad():
        test_outputs = linear_head(test_feats.to(args.device))
        test_preds = test_outputs.argmax(dim=1).cpu()
        test_acc = (test_preds == test_labels).float().mean().item()
        test_macro_f1 = f1_score(test_labels.numpy(), test_preds.numpy(), average="macro")
    print(f"Test metrics: acc={test_acc:.4f}, macro_f1={test_macro_f1:.4f}")

    import csv

    linear_head.eval()
    feature_loader = DataLoader(FeatureDataset(test_feats, test_labels, test_paths), batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for feats, labels_tensor, raw_paths in feature_loader:
            feats = feats.to(args.device)
            outputs = linear_head(feats)
            preds = outputs.argmax(dim=1).cpu().tolist()
            for pred, label, raw_rel in zip(preds, labels_tensor.tolist(), raw_paths):
                rows.append(
                    {
                        "image_path": raw_rel,
                        "pred_label": classes[pred],
                        "target_label": classes[label],
                    }
                )
    args.pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "pred_label", "target_label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

