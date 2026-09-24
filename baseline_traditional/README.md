# Traditional-motion baseline

This folder is the baseline corresponding to the CNN -> embedding -> PCA
pipeline. It uses the raw whole keypoint arrays, extracts one one-dimensional
signal per file with `traditional_motion.py`, extracts the same 11 kinematic
features, and runs the accepted z-score winsorized K-means (`z in [-1.5, 1.5]`).

The four existing CNN fold assignments are reused only to perform the same
fold-wise cluster-label-to-expert-score mapping. The raw clustering itself is
unsupervised and uses all files, matching the existing evaluation procedure.

Run from `final_code`:

```bash
python baseline_traditional/run_baseline.py --configs all
```

The final comparison is in `baseline_traditional/results/summary_metrics.csv`.
Each action also has `signals/`, `traditional_features.csv`, `cluster_labels.csv`,
`scores.csv`, and `metrics.csv`.

`precision`, `recall`, and `f1` are macro-averaged over expert scores 0--4;
`within1_accuracy` counts predictions whose absolute score error is at most 1.
