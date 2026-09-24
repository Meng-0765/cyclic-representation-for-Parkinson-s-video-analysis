# -*- coding: utf-8 -*-
"""Extract kinematic features from one-dimensional PCA signals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kinematic_features import extract_features


RESULTS_DIR = Path("results") / "cv_results_cnn"
PARTS = ("feet", "finger", "forearm", "palm", "toe")

FEATURE_COLUMNS = [
    "MeanFreq",
    "CovarFreq",
    "MeanVel",
    "CovarVel",
    "MeanAmp",
    "CovarAmp",
    "PeriodRange",
    "PrcInv",
    "Roughness",
    "DiffAmp",
    "DiffVel",
]


def process_action_folder(action_folder, fs=30, peak_params=None, force=False):
    pca_folder = action_folder / "PCA"
    output_folder = action_folder / "PCA_features"
    output_folder.mkdir(exist_ok=True)

    csv_path = output_folder / "pca_features.csv"
    npy_path = output_folder / "pca_features.npy"
    if csv_path.exists() and npy_path.exists() and not force:
        df = pd.read_csv(csv_path)
        return csv_path, npy_path, len(df)

    rows = []
    for pca_file in sorted(pca_folder.glob("*.npy")):
        signal = np.load(pca_file)
        features = extract_features(signal, fs=fs, peak_params=peak_params)
        rows.append({"file_name": pca_file.name, **features})

    df = pd.DataFrame(rows, columns=["file_name", *FEATURE_COLUMNS])
    df.to_csv(csv_path, index=False)
    np.save(npy_path, df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    return csv_path, npy_path, len(df)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract kinematic features from PCA arrays.")
    parser.add_argument("--configs", nargs="+", default=["all"], choices=["all", *PARTS])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--fs", type=float, default=30)
    parser.add_argument("--peak-distance", type=int, default=10)
    parser.add_argument("--peak-width", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute feature files even if outputs already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    peak_params = {"distance": args.peak_distance, "width": args.peak_width}
    configs = PARTS if "all" in args.configs else args.configs

    total = 0
    for config in configs:
        action_folder = args.results_dir / f"data_{config}"
        csv_path, npy_path, count = process_action_folder(
            action_folder,
            fs=args.fs,
            peak_params=peak_params,
            force=args.force,
        )
        total += count
        print(f"data_{config}: {count} files -> {csv_path} and {npy_path}")

    print(f"total_feature_rows={total}")


if __name__ == "__main__":
    main()
