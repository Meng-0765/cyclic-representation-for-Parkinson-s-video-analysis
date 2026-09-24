# -*- coding: utf-8 -*-
"""Run the traditional-signal baseline and compare it with expert scores.

Pipeline: raw whole keypoints -> traditional one-dimensional signal -> 11
kinematic features -> z-score winsorization at +/-1.5 -> K-means -> fold-wise
Spearman label mapping -> expert metrics.
"""

import argparse
import itertools
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kinematic_features import extract_features  # noqa: E402
from traditional_motion import extract_signal  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "baseline_traditional"
DATA_DIR = ROOT / "data" / "processed"
CNN_RESULTS_DIR = ROOT / "results" / "cv_results_cnn"
OUTPUT_DIR = BASELINE_DIR / "results"
PARTS = ("feet", "finger", "forearm", "palm", "toe")
CLASS_VALUES = [0, 1, 2, 3, 4]
FEATURE_COLUMNS = [
    "MeanFreq", "CovarFreq", "MeanVel", "CovarVel", "MeanAmp", "CovarAmp",
    "PeriodRange", "PrcInv", "Roughness", "DiffAmp", "DiffVel",
]


def parse_name(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)_(left|right)\.npy", name)
    if not match:
        raise ValueError(f"Unexpected file name: {name}")
    return int(match.group(1)), match.group(2)


def extract_outputs(action: str, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_dir = OUTPUT_DIR / f"data_{action}"
    signal_dir = action_dir / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    feature_path = action_dir / "traditional_features.csv"
    if feature_path.exists() and not force:
        return pd.read_csv(feature_path), action_dir
    rows = []
    for source in sorted((DATA_DIR / f"data_{action}_whole").glob("*.npy")):
        signal_path = signal_dir / source.name
        if signal_path.exists() and not force:
            signal = np.load(signal_path)
        else:
            data = np.load(source, allow_pickle=True)
            signal = extract_signal(f"data_{action}", data, source.name)
            np.save(signal_path, signal.astype(np.float32))
        feats = extract_features(signal, fs=30, peak_params={"distance": 10, "width": 3})
        rows.append({"file_name": source.name, **feats})
    features = pd.DataFrame(rows, columns=["file_name", *FEATURE_COLUMNS])
    if force or not feature_path.exists():
        features.to_csv(feature_path, index=False)
    else:
        features = pd.read_csv(feature_path)
    return features, action_dir


def fold_lookup(action: str) -> dict[str, int]:
    lookup = {}
    for fold in range(1, 5):
        folder = CNN_RESULTS_DIR / f"data_{action}" / f"fold_{fold}" / "embeddings_whole"
        for path in folder.glob("*.npy"):
            if path.name in lookup:
                raise ValueError(f"{path.name} occurs in multiple folds.")
            lookup[path.name] = fold
    return lookup


def zscore_winsorize(features: pd.DataFrame) -> np.ndarray:
    raw = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(raw.median(numeric_only=True))
    return np.clip(StandardScaler().fit_transform(raw), -1.5, 1.5)


def spearman_mapping(train: pd.DataFrame) -> dict[int, int]:
    clusters = sorted(set(CLASS_VALUES) | set(train["cluster"].astype(int)))
    best = None
    for values in itertools.permutations(CLASS_VALUES, len(clusters)):
        mapping = dict(zip(clusters, values))
        predicted = train["cluster"].map(mapping).astype(int)
        corr = spearmanr(train["expert_score"], predicted).statistic
        corr = -np.inf if pd.isna(corr) else float(corr)
        exact = float((predicted == train["expert_score"]).mean())
        mae = float((predicted - train["expert_score"]).abs().mean())
        candidate = (corr, exact, -mae, tuple(-mapping[c] for c in clusters), mapping)
        if best is None or candidate > best:
            best = candidate
    return best[-1]


def run_action(action: str, expert: pd.DataFrame, force: bool) -> dict:
    features, action_dir = extract_outputs(action, force)
    scores = expert[expert["action"].eq(action)][["file_name", "id", "side", "expert_score"]].copy()
    scores = scores.merge(features, on="file_name", how="inner", validate="one_to_one")
    scores["fold"] = scores["file_name"].map(fold_lookup(action))
    if scores["fold"].isna().any():
        raise ValueError(f"Missing fold assignment for data_{action}.")

    x = zscore_winsorize(scores)
    cluster = KMeans(n_clusters=5, random_state=42, n_init=100).fit_predict(x)
    scores["cluster"] = cluster
    tested = []
    for test_fold in range(1, 5):
        train = scores[scores["fold"] != test_fold]
        test = scores[scores["fold"] == test_fold].copy()
        mapping = spearman_mapping(train)
        test["algorithm_score"] = test["cluster"].map(mapping).astype(int)
        test["test_exact_match"] = test["algorithm_score"].eq(test["expert_score"])
        test["within1_match"] = (test["algorithm_score"] - test["expert_score"]).abs().le(1)
        test["test_absolute_error"] = (test["algorithm_score"] - test["expert_score"]).abs()
        test["test_fold"] = test_fold
        tested.append(test)

    detail = pd.concat(tested, ignore_index=True).sort_values(["fold", "id", "side"])
    detail = detail[["file_name", "id", "side", "fold", "test_fold", "expert_score",
                     "algorithm_score", "test_exact_match", "within1_match",
                     "test_absolute_error", "cluster"]]
    y_true = detail["expert_score"].astype(int)
    y_pred = detail["algorithm_score"].astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_VALUES, average="macro", zero_division=0
    )
    metrics = {
        "action": action,
        "n": len(detail),
        "within1_accuracy": float(detail["within1_match"].mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_accuracy": float(detail["test_exact_match"].mean()),
        "mae": float(detail["test_absolute_error"].mean()),
    }
    action_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(action_dir / "scores.csv", index=False)
    features.to_csv(action_dir / "traditional_features.csv", index=False)
    pd.DataFrame([metrics]).to_csv(action_dir / "metrics.csv", index=False)
    pd.DataFrame({"file_name": scores["file_name"], "cluster": cluster}).to_csv(
        action_dir / "cluster_labels.csv", index=False
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["all"], choices=["all", *PARTS])
    parser.add_argument("--expert-csv", type=Path, default=BASELINE_DIR / "expert_scores.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = PARTS if "all" in args.configs else args.configs
    expert = pd.read_csv(args.expert_csv)
    metrics = [run_action(action, expert, args.force) for action in configs]
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
