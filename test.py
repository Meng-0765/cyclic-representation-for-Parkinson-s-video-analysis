# -*- coding: utf-8 -*-
"""Export whole-sequence embeddings with a trained fold model."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

from train import (
    PARTS,
    PROCESSED_DIR,
    RESULTS_DIR,
    KeypointEmbeddingCNN,
    collect_files_by_ids,
    get_input_joint_count,
    list_npy_files,
    parse_sample_id,
)


def export_embeddings(model, file_paths, output_dir, device, config_name):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export whole-file embeddings with a trained CNN model.")
    parser.add_argument("--config", required=True, choices=PARTS)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ids", nargs="*", help="Optional numeric IDs to export. Defaults to all whole files.")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--use-attention", action="store_true", help="Use only for checkpoints trained with TemporalAttention.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = args.input_dir or PROCESSED_DIR / f"data_{args.config}_whole"
    output_dir = args.output_dir or RESULTS_DIR / f"data_{args.config}" / "manual_test_embeddings_whole"

    file_paths = collect_files_by_ids(input_dir, set(args.ids)) if args.ids else list_npy_files(input_dir)
    model = KeypointEmbeddingCNN(
        num_joints=get_input_joint_count(args.config),
        smoothing_window=args.smoothing_window,
        use_attention=args.use_attention,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    export_embeddings(model, file_paths, output_dir, device, args.config)

    exported_ids = {parse_sample_id(path.name) for path in file_paths}
    print(f"device={device}")
    print(f"config={args.config}")
    print(f"model={args.model_path}")
    print(f"input_files={len(file_paths)} ids={len(exported_ids)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
