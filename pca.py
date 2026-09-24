# -*- coding: utf-8 -*-
"""Collect fold embeddings and export one-dimensional PCA arrays."""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


RESULTS_DIR = Path("results") / "cv_results_cnn"
PARTS = ("feet", "finger", "forearm", "palm", "toe")


def compress_embedding_pca_1d(embedding):
    pca = PCA(n_components=1)
    compressed = pca.fit_transform(embedding)
    return compressed[:, 0], float(pca.explained_variance_ratio_[0])


def collect_fold_embeddings(base_dir, combined_embedding_dir, folds=4):
    combined_embedding_dir.mkdir(parents=True, exist_ok=True)
    fold_files = []
    seen = set()

    for fold_index in range(1, folds + 1):
        embedding_dir = base_dir / f"fold_{fold_index}" / "embeddings_whole"
        if not embedding_dir.exists():
            raise FileNotFoundError(f"Missing embedding directory: {embedding_dir}")

        for source_path in sorted(embedding_dir.glob("*.npy")):
            if source_path.name in seen:
                raise RuntimeError(f"Duplicate test embedding across folds: {source_path.name}")
            seen.add(source_path.name)

            target_path = combined_embedding_dir / source_path.name
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
            fold_files.append((fold_index, target_path))

    return fold_files


def export_pca(fold_files, pca_dir, summary_path, force=False):
    pca_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold_index, embedding_path in fold_files:
        embedding = np.load(embedding_path)
        pca_path = pca_dir / embedding_path.name
        pca_1d, explained_variance_ratio = compress_embedding_pca_1d(embedding)
        if pca_path.exists() and not force:
            saved_pca = np.load(pca_path)
            if saved_pca.shape != pca_1d.shape:
                raise RuntimeError(f"Existing PCA shape mismatch: {pca_path}")
            pca_1d = saved_pca
        else:
            np.save(pca_path, pca_1d)

        rows.append(
            {
                "file_name": embedding_path.name,
                "fold": fold_index,
                "embedding_length": embedding.shape[0],
                "embedding_dim": embedding.shape[1],
                "pca_length": pca_1d.shape[0],
                "explained_variance_ratio": explained_variance_ratio,
                "embedding_path": str(embedding_path),
                "pca_path": str(pca_path),
            }
        )

    if rows and (force or not summary_path.exists()):
        try:
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        except PermissionError:
            print(f"Warning: cannot write optional PCA summary: {summary_path}")

    return rows


def process_base_dir(base_dir, folds=4, force=False):
    combined_embedding_dir = base_dir / "all_test_embeddings_whole"
    pca_dir = base_dir / "PCA"
    summary_path = base_dir / "pca_summary.csv"

    fold_files = collect_fold_embeddings(base_dir, combined_embedding_dir, folds=folds)
    rows = export_pca(fold_files, pca_dir, summary_path, force=force)
    print(f"[{base_dir.name}] combined_embeddings={len(fold_files)} dir={combined_embedding_dir}")
    print(f"[{base_dir.name}] pca_files={len(rows)} dir={pca_dir}")
    print(f"[{base_dir.name}] summary={summary_path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Collect fold embeddings and export 1D PCA arrays.")
    parser.add_argument("--configs", nargs="+", default=["all"], choices=["all", *PARTS])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--base-dir", type=Path, help="Optional single result directory, for example results/cv_results_cnn/data_palm.")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Recompute PCA arrays even when they already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.base_dir:
        process_base_dir(args.base_dir, folds=args.folds, force=args.force)
        return

    configs = PARTS if "all" in args.configs else args.configs
    total = 0
    for config in configs:
        total += len(process_base_dir(args.results_dir / f"data_{config}", folds=args.folds, force=args.force))
    print(f"total_pca_files={total}")


if __name__ == "__main__":
    main()
