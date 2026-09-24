# -*- coding: utf-8 -*-
"""Run z-score winsorized K-means clustering on PCA kinematic features.

This is the clustering pipeline used by winsorize_zscore_kmeans_spearman_z1_5.py:
features -> StandardScaler z-scores -> clip to [-1.5, 1.5] -> K-means.
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = Path("results") / "cv_results_cnn"
PARTS = ("feet", "finger", "forearm", "palm", "toe")
DEFAULT_OUTPUT_DIR_NAME = "clustering_results"

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


def zscore_winsorize(feature_df, z_clip=1.5):
    raw = feature_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    raw = raw.fillna(raw.median(numeric_only=True))

    z_values = StandardScaler().fit_transform(raw.to_numpy(dtype=float))
    z_df = pd.DataFrame(z_values, columns=FEATURE_COLUMNS)
    clipped_z_df = z_df.clip(lower=-z_clip, upper=z_clip)

    clipping_mask = z_df.ne(clipped_z_df)
    feature_summary = []
    for col in FEATURE_COLUMNS:
        feature_summary.append(
            {
                "feature": col,
                "n_clipped_low": int((z_df[col] < -z_clip).sum()),
                "n_clipped_high": int((z_df[col] > z_clip).sum()),
                "n_clipped_total": int(clipping_mask[col].sum()),
                "percent_clipped_total": float(clipping_mask[col].mean()),
                "min_z_before": float(z_df[col].min()),
                "max_z_before": float(z_df[col].max()),
            }
        )

    sample_audit = feature_df[["file_name"]].copy()
    sample_audit["n_features_clipped"] = clipping_mask.sum(axis=1).astype(int)
    sample_audit["was_any_feature_clipped"] = sample_audit["n_features_clipped"] > 0
    sample_audit["clipped_features"] = clipping_mask.apply(
        lambda row: ",".join(row.index[row].tolist()),
        axis=1,
    )
    sample_audit["global_z_norm_before_clip"] = np.linalg.norm(z_df.to_numpy(dtype=float), axis=1)
    sample_audit["global_z_norm_after_clip"] = np.linalg.norm(clipped_z_df.to_numpy(dtype=float), axis=1)
    abs_z = z_df.abs()
    sample_audit["top_abs_z_feature_before_clip"] = abs_z.idxmax(axis=1)
    sample_audit["top_abs_z_value_before_clip"] = abs_z.max(axis=1)

    return clipped_z_df, pd.DataFrame(feature_summary), sample_audit


def run_winsorized_kmeans(
    action_folder,
    n_clusters=5,
    random_state=42,
    z_clip=1.5,
    output_dir_name=DEFAULT_OUTPUT_DIR_NAME,
    force=False,
):
    action_folder = Path(action_folder)
    output_folder = action_folder / output_dir_name
    labels_path = output_folder / "cluster_labels.csv"
    metrics_path = output_folder / "clustering_metrics.csv"
    if labels_path.exists() and metrics_path.exists() and not force:
        labels = pd.read_csv(labels_path)
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        return labels, metrics

    feature_path = action_folder / "PCA_features" / "pca_features.csv"
    df = pd.read_csv(feature_path)
    x_clipped, feature_summary, sample_audit = zscore_winsorize(df, z_clip=z_clip)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=100)
    labels = model.fit_predict(x_clipped.to_numpy(dtype=float))

    output_folder.mkdir(exist_ok=True)
    label_df = pd.DataFrame({"file_name": df["file_name"], "K-means_cluster": labels})
    label_df.to_csv(labels_path, index=False)

    metrics = {
        "Action": action_folder.name,
        "Method": f"StandardScaler z-score winsorized to +/-{z_clip} + K-means",
        "z_clip": z_clip,
        "N": int(len(df)),
        "n_samples_any_feature_clipped": int(sample_audit["was_any_feature_clipped"].sum()),
        "percent_samples_any_feature_clipped": float(sample_audit["was_any_feature_clipped"].mean()),
        "n_clusters": n_clusters,
        "Silhouette": float(silhouette_score(x_clipped, labels)),
        "DBI": float(davies_bouldin_score(x_clipped, labels)),
        "CH": float(calinski_harabasz_score(x_clipped, labels)),
        "Inertia": float(model.inertia_),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    return label_df, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run z-score winsorized K-means on PCA feature CSV files.")
    parser.add_argument("--configs", nargs="+", default=["all"], choices=["all", *PARTS])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--z-clip", type=float, default=1.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--force", action="store_true", help="Recompute labels even if outputs already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = PARTS if "all" in args.configs else args.configs
    for config in configs:
        action_folder = args.results_dir / f"data_{config}"
        labels, metrics = run_winsorized_kmeans(
            action_folder,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            z_clip=args.z_clip,
            output_dir_name=args.output_dir_name,
            force=args.force,
        )
        print(
            f"data_{config}: {len(labels)} labels -> "
            f"{action_folder / args.output_dir_name / 'cluster_labels.csv'}; "
            f"silhouette={metrics['Silhouette']:.4f}"
        )


if __name__ == "__main__":
    main()
