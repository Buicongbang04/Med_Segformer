import os
import glob
import argparse
import random

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


def normalize_brats_flair(volume):
    """
    BraTS MRI FLAIR preprocessing:
    - foreground = intensity > 0
    - percentile clipping on foreground only
    - z-score using foreground only
    - clip [-5, 5]
    - rescale to [0, 1]
    - background set back to 0
    """
    volume = volume.astype(np.float32)
    foreground = volume > 0

    if foreground.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32)

    fg_values = volume[foreground]

    p1, p99 = np.percentile(fg_values, [1, 99])
    volume_clipped = volume.copy()
    volume_clipped[foreground] = np.clip(volume_clipped[foreground], p1, p99)

    fg_values = volume_clipped[foreground]
    mean = fg_values.mean()
    std = fg_values.std()

    if std < 1e-6:
        std = 1.0

    volume_norm = (volume_clipped - mean) / std
    volume_norm = np.clip(volume_norm, -5, 5)

    volume_norm = (volume_norm + 5) / 10.0
    volume_norm[~foreground] = 0.0

    return volume_norm.astype(np.float32)


def save_case_slices(case_dir, out_img_dir, out_mask_dir, skip_empty=True):
    case_name = os.path.basename(case_dir)

    flair_paths = glob.glob(os.path.join(case_dir, "*_flair.nii"))
    seg_paths = glob.glob(os.path.join(case_dir, "*_seg.nii"))

    if len(flair_paths) != 1 or len(seg_paths) != 1:
        print(f"[SKIP] {case_name}: cannot find flair/seg")
        return 0

    flair_path = flair_paths[0]
    seg_path = seg_paths[0]

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    flair = normalize_brats_flair(flair)
    mask = (seg > 0).astype(np.uint8)

    num_saved = 0

    # BraTS shape is usually [H, W, Z]
    for z in range(flair.shape[2]):
        img_slice = flair[:, :, z]
        mask_slice = mask[:, :, z]

        if skip_empty and mask_slice.sum() == 0:
            # keep fewer empty slices to reduce imbalance
            if random.random() > 0.15:
                continue

        img_uint8 = (img_slice * 255).astype(np.uint8)
        img_3ch = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

        mask_uint8 = mask_slice.astype(np.uint8)

        filename = f"{case_name}_z{z:03d}.png"

        cv2.imwrite(os.path.join(out_img_dir, filename), img_3ch)
        cv2.imwrite(os.path.join(out_mask_dir, filename), mask_uint8)

        num_saved += 1

    return num_saved


def split_cases(case_dirs, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    case_dirs = sorted(case_dirs)
    random.shuffle(case_dirs)

    n = len(case_dirs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = case_dirs[:n_train]
    val_cases = case_dirs[n_train:n_train + n_val]
    test_cases = case_dirs[n_train + n_val:]

    return train_cases, val_cases, test_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", default="data/BraTS20/MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--out-root", default="data/medical_seg/brain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-empty", action="store_true")
    args = parser.parse_args()

    case_dirs = [
        d for d in glob.glob(os.path.join(args.brats_root, "*"))
        if os.path.isdir(d)
    ]

    train_cases, val_cases, test_cases = split_cases(
        case_dirs,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed
    )

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }

    for split, cases in splits.items():
        img_dir = os.path.join(args.out_root, "images", split)
        mask_dir = os.path.join(args.out_root, "annotations", split)

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        total = 0
        for case_dir in tqdm(cases, desc=f"Processing {split}"):
            total += save_case_slices(
                case_dir,
                img_dir,
                mask_dir,
                skip_empty=not args.keep_empty
            )

        print(f"{split}: {len(cases)} cases, {total} slices saved")


if __name__ == "__main__":
    main()
