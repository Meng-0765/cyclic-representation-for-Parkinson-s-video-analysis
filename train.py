# -*- coding: utf-8 -*-
"""
Train CNN keypoint embedding models with ID-level cross validation.

Segmented files are used for training:
    data/processed/data_<part>/*.npy

Whole, unsegmented files are used for fold testing/export:
    data/processed/data_<part>_whole/*.npy
"""

import argparse
import os
import random
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROCESSED_DIR = Path("data") / "processed"
RESULTS_DIR = Path("results") / "cv_results_cnn"

PARTS = ("feet", "finger", "forearm", "palm", "toe")

# Input files are already prepared with the complete point set for each
# task. Hand-related tasks use one side of the hand (21 points), while foot
# tasks use the full pose keypoints (25 points). The CNN combines the two
# coordinates of each point into its channel dimension, yielding 42 and 50
# input channels respectively. No task-specific joint indexing/slicing is
# performed in the dataset or export pipeline.
INPUT_JOINT_COUNTS = {
    "finger": 21,
    "palm": 21,
    "forearm": 21,
    "toe": 25,
    "feet": 25,
}


def get_input_joint_count(config_name):
    try:
        return INPUT_JOINT_COUNTS[config_name]
    except KeyError as exc:
        raise ValueError(f"Unknown config name: {config_name}") from exc


def parse_sample_id(file_name):
    match = re.match(r"^(\d+)_", file_name)
    if not match:
        raise ValueError(f"Cannot parse sample id from file name: {file_name}")
    return match.group(1)


def list_npy_files(data_dir):
    return sorted(path for path in Path(data_dir).iterdir() if path.suffix == ".npy")


def build_id_folds(ids, n_splits=4, seed=42):
    ids = sorted(ids, key=lambda value: int(value))
    rng = random.Random(seed)
    rng.shuffle(ids)

    fold_sizes = [len(ids) // n_splits] * n_splits
    for i in range(len(ids) % n_splits):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_ids = set(ids[start:stop])
        train_ids = set(ids) - test_ids
        folds.append((train_ids, test_ids))
        start = stop
    return folds


def collect_files_by_ids(data_dir, ids):
    ids = set(ids)
    return [path for path in list_npy_files(data_dir) if parse_sample_id(path.name) in ids]


def split_train_validation_ids(train_ids, validation_fraction=0.2, seed=42):
    """Split fold-training IDs into disjoint train/validation ID sets."""
    ids = sorted(set(train_ids), key=lambda value: int(value))
    if len(ids) < 2 or validation_fraction <= 0:
        return set(ids), set()

    rng = random.Random(seed)
    rng.shuffle(ids)
    n_validation = max(1, int(round(len(ids) * validation_fraction)))
    n_validation = min(n_validation, len(ids) - 1)
    validation_ids = set(ids[:n_validation])
    training_ids = set(ids[n_validation:])
    return training_ids, validation_ids


class KeypointDataset(Dataset):
    def __init__(self, file_paths, config_name):
        self.file_paths = list(file_paths)
        self.config_name = config_name
        self.expected_joints = get_input_joint_count(config_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        keypoints = np.load(self.file_paths[idx])
        if keypoints.ndim != 3 or keypoints.shape[2] != 2:
            raise ValueError(
                f"Expected {self.file_paths[idx].name} to have shape [T, J, 2], "
                f"got {keypoints.shape}"
            )
        if keypoints.shape[1] != self.expected_joints:
            raise ValueError(
                f"Config '{self.config_name}' expects {self.expected_joints} input joints, "
                f"but {self.file_paths[idx].name} has {keypoints.shape[1]}"
            )
        return torch.tensor(keypoints, dtype=torch.float32)


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, 1)

    def forward(self, x):
        attn = F.relu(self.fc1(x))
        attn = self.fc2(attn)
        return x * (1 + attn)


class KeypointEmbeddingCNN(nn.Module):
    def __init__(self, num_joints, output_dim=128, smoothing_window=5, use_attention=False):
        super().__init__()
        in_channels = num_joints * 2

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, output_dim, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.smoothing_window = smoothing_window
        self.temporal_attention = TemporalAttention(output_dim) if use_attention else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        kernel = torch.ones(1, 1, self.smoothing_window, device=x.device) / self.smoothing_window
        x = F.conv1d(
            x,
            kernel.expand(x.size(1), -1, -1),
            groups=x.size(1),
            padding=self.smoothing_window // 2,
        )

        x = x.permute(0, 2, 1)
        return self.temporal_attention(x)


def compute_tsm(embeddings):
    batch_size, seq_len, _ = embeddings.shape
    tsm = torch.zeros((batch_size, seq_len, seq_len), device=embeddings.device)
    for i in range(batch_size):
        tsm[i] = -torch.cdist(embeddings[i], embeddings[i], p=2) ** 2
    return tsm


def sample_triplets_topk(tsm, top_k=10, num_anchors=3):
    batch_size, seq_len, _ = tsm.shape
    anchors, positives, negatives = [], [], []

    for i in range(batch_size):
        a_list, p_list, n_list = [], [], []
        anchor_idx = torch.randperm(seq_len, device=tsm.device)[:num_anchors]

        for anchor in anchor_idx:
            similarities = tsm[i, anchor]
            sorted_idx = torch.argsort(similarities, descending=True)
            pos_idx = sorted_idx[sorted_idx != anchor][:top_k]
            neg_idx = sorted_idx[-top_k:]
            p_list.append(similarities[pos_idx])
            n_list.append(similarities[neg_idx])
            a_list.append(similarities[anchor].unsqueeze(0))

        anchors.append(a_list)
        positives.append(p_list)
        negatives.append(n_list)

    return anchors, positives, negatives


class TripletLoss(nn.Module):
    def __init__(self, margin=0.8):
        super().__init__()
        self.margin = margin

    def forward(self, anchors, positives, negatives):
        losses = []
        for a_list, p_list, n_list in zip(anchors, positives, negatives):
            for anchor, positive, negative in zip(a_list, p_list, n_list):
                losses.append(F.relu(negative.sum() - positive.sum() + self.margin))
        return torch.stack(losses).mean()


def train_step(model, optimizer, batch, top_k, num_anchors, margin):
    optimizer.zero_grad()
    embeddings = F.normalize(model(batch), p=2, dim=2)
    tsm = compute_tsm(embeddings)
    anchors, positives, negatives = sample_triplets_topk(tsm, top_k=top_k, num_anchors=num_anchors)
    loss = TripletLoss(margin=margin)(anchors, positives, negatives)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def train_epoch(dataloader, model, optimizer, device, args):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        batch = batch.to(device)
        total_loss += train_step(model, optimizer, batch, args.top_k, args.num_anchors, args.margin)
        num_batches += 1
        if args.max_batches and num_batches >= args.max_batches:
            break
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validation_epoch(dataloader, model, device, args):
    """Compute the triplet loss without updating model parameters."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_fn = TripletLoss(margin=args.margin)
    for batch in dataloader:
        batch = batch.to(device)
        embeddings = F.normalize(model(batch), p=2, dim=2)
        tsm = compute_tsm(embeddings)
        anchors, positives, negatives = sample_triplets_topk(
            tsm, top_k=args.top_k, num_anchors=args.num_anchors
        )
        total_loss += loss_fn(anchors, positives, negatives).item()
        num_batches += 1
        if args.max_batches and num_batches >= args.max_batches:
            break
    return total_loss / max(num_batches, 1)


def save_learning_curve(history, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    curve_csv = output_path.with_suffix(".csv")
    np.savetxt(
        curve_csv,
        np.column_stack((epochs, history["train_loss"], history["validation_loss"])),
        delimiter=",",
        header="epoch,training_loss,validation_loss",
        comments="",
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["validation_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def export_whole_embeddings(model, file_paths, output_dir, device, config_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_joints = get_input_joint_count(config_name)
    model.eval()

    with torch.no_grad():
        for file_path in file_paths:
            data = np.load(file_path)
            if data.ndim != 3 or data.shape[2] != 2:
                raise ValueError(f"Expected {file_path.name} to have shape [T, J, 2], got {data.shape}")
            if data.shape[1] != expected_joints:
                raise ValueError(
                    f"Config '{config_name}' expects {expected_joints} input joints, "
                    f"but {file_path.name} has {data.shape[1]}"
                )
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            embeddings = F.normalize(model(data_tensor), p=2, dim=2)[0]
            np.save(output_dir / file_path.name, embeddings.cpu().numpy())


def run_fold(part, fold_index, train_ids, test_ids, args, device):
    train_dir = args.data_dir / f"data_{part}"
    whole_dir = args.data_dir / f"data_{part}_whole"
    result_dir = args.output_dir / f"data_{part}" / f"fold_{fold_index}"
    model_dir = result_dir / "model"
    embedding_dir = result_dir / "embeddings_whole"

    training_ids, validation_ids = split_train_validation_ids(
        train_ids, validation_fraction=args.validation_fraction, seed=args.seed + fold_index
    )
    train_files = collect_files_by_ids(train_dir, training_ids)
    validation_files = collect_files_by_ids(train_dir, validation_ids)
    test_whole_files = collect_files_by_ids(whole_dir, test_ids)
    if set(train_ids) & set(test_ids) or training_ids & validation_ids:
        raise RuntimeError(f"ID leakage in fold {fold_index}")

    print(
        f"[{part} fold {fold_index}] "
        f"train_ids={len(training_ids)} validation_ids={len(validation_ids)} "
        f"test_ids={len(test_ids)} train_segments={len(train_files)} "
        f"validation_segments={len(validation_files)} test_whole={len(test_whole_files)}"
    )
    if args.dry_run:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    dataset = KeypointDataset(train_files, config_name=part)
    validation_dataset = KeypointDataset(validation_files, config_name=part)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = KeypointEmbeddingCNN(
        num_joints=get_input_joint_count(part),
        smoothing_window=args.smoothing_window,
        use_attention=args.use_attention,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(dataloader, model, optimizer, device, args)
        validation_loss = validation_epoch(validation_dataloader, model, device, args)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"[{part} fold {fold_index}] Epoch {epoch + 1}/{args.epochs}, "
            f"Training loss: {train_loss:.7f}, Validation loss: {validation_loss:.7f}"
        )
        if validation_loss < best_validation_loss - args.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.save_every_epoch:
            torch.save(model.state_dict(), model_dir / f"model_epoch_{epoch + 1}.pth")
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"[{part} fold {fold_index}] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_learning_curve(
        history,
        result_dir / "learning_curve.png",
        f"{part} fold {fold_index}: training and validation loss",
    )

    torch.save(model.state_dict(), model_dir / "model_final.pth")
    export_whole_embeddings(model, test_whole_files, embedding_dir, device, part)


def run_part(part, args, device):
    train_dir = args.data_dir / f"data_{part}"
    whole_dir = args.data_dir / f"data_{part}_whole"
    train_ids = {parse_sample_id(path.name) for path in list_npy_files(train_dir)}
    whole_ids = {parse_sample_id(path.name) for path in list_npy_files(whole_dir)}
    ids = sorted(train_ids & whole_ids, key=lambda value: int(value))

    print(f"[{part}] usable_ids={len(ids)}")
    folds = build_id_folds(ids, n_splits=args.folds, seed=args.seed)
    for fold_index, (train_ids_fold, test_ids_fold) in enumerate(folds, start=1):
        run_fold(part, fold_index, train_ids_fold, test_ids_fold, args, device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/test CNN embeddings with ID-level CV.")
    parser.add_argument("--configs", nargs="+", default=["all"], choices=["all", *PARTS])
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-anchors", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="Debug only: limit batches per epoch.")
    parser.add_argument("--dry-run", action="store_true", help="Print folds without training.")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--use-attention", action="store_true", help="Use TemporalAttention. Existing shared results use the default off setting.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts = PARTS if "all" in args.configs else tuple(args.configs)

    print(f"device={device}")
    print(f"data_dir={args.data_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"model=CNN use_attention={args.use_attention}")
    for part in parts:
        run_part(part, args, device)


if __name__ == "__main__":
    main()
