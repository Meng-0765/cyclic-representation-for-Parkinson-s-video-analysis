# CNN Keypoint Embedding Cross Validation

This folder contains a compact reproducible pipeline for CNN model training/testing, one-dimensional PCA extraction, kinematic feature extraction, and final z=1.5 winsorized K-means clustering.

## Folder Layout

```text
final_code/
  train.py
  test.py
  pca.py
  kinematic_features.py
  extract_pca_features.py
  kmeans_cluster.py
  id_mapping.csv
  baseline_traditional/
  requirements.txt
  data/                 # intentionally empty: raw keypoint input is excluded
  results/              # disclosed intermediate and final results
```

The original keypoint data cannot be disclosed and is therefore excluded from `data/`. The generated model checkpoints, embeddings, PCA arrays, kinematic features, clustering outputs, and traditional-signal baseline outputs are retained because they are permitted intermediate or final results. `train.py` and `test.py` are included as reference/reproduction scripts, but cannot run without the original input data in a separate authorized local copy.

When authorized data are available, segmented folders such as `data/processed/data_palm/` are used for training. Whole folders such as `data/processed/data_palm_whole/` are used for testing/exporting embeddings. File names are grouped by the numeric subject/sample ID before the first underscore, so `52_left0.npy`, `52_left1.npy`, `52_left.npy`, and `52_right.npy` stay in the same fold.

The downstream scripts can run from the disclosed intermediate results: embeddings are consumed by `pca.py`, one-dimensional PCA arrays by `extract_pca_features.py`, and the feature table by `kmeans_cluster.py`.

## Install

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA/CPU environment if the default wheel is not suitable.

## Training And Testing With Authorized Data

Run all five parts with four-fold ID-level cross validation:

```bash
python train.py --configs all --epochs 30 --batch-size 32
```

Run only one part:

```bash
python train.py --configs palm --epochs 30 --batch-size 32
```

Check the fold split without training:

```bash
python train.py --configs all --dry-run
```

The training script saves each fold model to:

```text
results/cv_results_cnn/data_<part>/fold_<n>/model/model_final.pth
```

It also exports whole-sequence test embeddings to:

```text
results/cv_results_cnn/data_<part>/fold_<n>/embeddings_whole/
```

## Export Embeddings With A Saved Model

```bash
python test.py --config palm --model-path results/cv_results_cnn/data_palm/fold_1/model/model_final.pth
```

By default, this processes all files in `data/processed/data_palm_whole/`. Use `--ids` to export selected IDs only.

## Export PCA Data

The PCA step does not save images. It saves one `.npy` file per embedding, containing a one-dimensional PCA signal.

```bash
python pca.py --configs all
```

Outputs are saved under each part:

```text
results/cv_results_cnn/data_<part>/all_test_embeddings_whole/
results/cv_results_cnn/data_<part>/PCA/
```

When recomputed, `pca.py` also attempts to write `pca_summary.csv` as a small provenance table. This summary is optional and is not required by later steps.

## Extract Kinematic Features

After PCA data exists, extract the motion/kinematic feature table:

```bash
python extract_pca_features.py --configs all
```

This reads:

```text
results/cv_results_cnn/data_<part>/PCA/
```

and writes:

```text
results/cv_results_cnn/data_<part>/PCA_features/pca_features.csv
results/cv_results_cnn/data_<part>/PCA_features/pca_features.npy
```

The feature columns are:

```text
MeanFreq, CovarFreq, MeanVel, CovarVel, MeanAmp, CovarAmp,
PeriodRange, PrcInv, Roughness, DiffAmp, DiffVel
```

## Clustering

The main clustering pipeline follows `winsorize_zscore_kmeans_spearman_z1_5.py`:

```text
PCA_features -> StandardScaler z-scores -> winsorize to [-1.5, 1.5] -> K-means
```

Run the winsorized K-means clustering:

```bash
python kmeans_cluster.py --configs all --n-clusters 5
```

This writes:

```text
results/cv_results_cnn/data_<part>/clustering_results/cluster_labels.csv
results/cv_results_cnn/data_<part>/clustering_results/clustering_metrics.csv
```

Pass `--force` to recompute existing local outputs.

## Notes

The distributed checkpoints and results are generated artifacts; they do not contain the original keypoint input. The default CNN architecture in `train.py` has temporal attention disabled. Use `--use-attention` only if you train and test locally generated checkpoints with attention enabled.

The default behavior of `pca.py`, `extract_pca_features.py`, and `kmeans_cluster.py` is non-destructive: when included outputs already exist, they are reused. Pass `--force` to recompute them.

Raw keypoint data is intentionally not included here. Embeddings, PCA files, feature tables, baseline outputs, and the final z=1.5 clustering result under `clustering_results/` are included.

All selected sample IDs are remapped to a contiguous sequence starting at 1.
The same mapping is applied to segmented files, whole files, embeddings, PCA
arrays, feature tables, clustering tables, and baseline scores. The original
to new ID lookup is stored in `id_mapping.csv` for provenance.
