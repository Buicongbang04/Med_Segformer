import json
import os
import glob
import argparse
import random
import shutil

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXT = ".png"


def normalize_brats_flair(volume):
    """
    BraTS MRI FLAIR preprocessing.

    Medical-safe rule:
    - foreground = intensity > 0
    - percentile clipping on foreground only
    - z-score using foreground only
    - clip [-5, 5]
    - rescale to [0, 1]
    - background set back to 0

    This is 3D volume-level normalization, NOT per-slice normalization.
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


def find_single_file(case_dir, suffix):
    paths = glob.glob(os.path.join(case_dir, f"*{suffix}"))
    if len(paths) != 1:
        return None
    return paths[0]


def should_keep_slice(split, has_tumor, keep_empty_all, empty_keep_prob):
    """
    Slice keeping policy.

    Current project decision:
    - train/val: can keep old 2D behavior for MMSeg mDice monitoring
    - test: MUST keep all slices for valid 3D Dice and tumor volume estimation
    """
    if split == "test":
        return True

    if keep_empty_all:
        return True

    if has_tumor:
        return True

    return random.random() < empty_keep_prob


def save_case_slices(
    case_dir,
    out_img_dir,
    out_mask_dir,
    split,
    keep_empty_all=False,
    empty_keep_prob=0.15,
):
    case_name = os.path.basename(case_dir)

    flair_path = find_single_file(case_dir, "_flair.nii")
    seg_path = find_single_file(case_dir, "_seg.nii")

    if flair_path is None or seg_path is None:
        print(f"[SKIP] {case_name}: cannot find exactly one flair/seg")
        return [], 0

    flair_nii = nib.load(flair_path)
    seg_nii = nib.load(seg_path)

    flair = flair_nii.get_fdata()
    seg = seg_nii.get_fdata()

    if flair.shape != seg.shape:
        print(f"[SKIP] {case_name}: flair/seg shape mismatch {flair.shape} vs {seg.shape}")
        return [], 0

    flair = normalize_brats_flair(flair)
    mask = (seg > 0).astype(np.uint8)

    rows = []
    num_saved = 0
    depth = flair.shape[2]

    for z in range(depth):
        img_slice = flair[:, :, z]
        mask_slice = mask[:, :, z]

        tumor_voxels = int(mask_slice.sum())
        has_tumor = tumor_voxels > 0

        kept = should_keep_slice(
            split=split,
            has_tumor=has_tumor,
            keep_empty_all=keep_empty_all,
            empty_keep_prob=empty_keep_prob,
        )

        filename = f"{case_name}_z{z:03d}{IMAGE_EXT}"
        img_path = os.path.join(out_img_dir, filename)
        mask_path = os.path.join(out_mask_dir, filename)

        if kept:
            img_uint8 = np.clip(img_slice * 255.0, 0, 255).astype(np.uint8)
            img_3ch = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

            # MMSeg CustomDataset expects class indices, not 0/255 binary mask.
            mask_uint8 = mask_slice.astype(np.uint8)

            cv2.imwrite(img_path, img_3ch)
            cv2.imwrite(mask_path, mask_uint8)
            num_saved += 1
        else:
            img_path = ""
            mask_path = ""

        rows.append({
            "split": split,
            "case_id": case_name,
            "slice_index": z,
            "depth": depth,
            "image_path": img_path,
            "mask_path": mask_path,
            "has_tumor": int(has_tumor),
            "tumor_voxels": tumor_voxels,
            "kept": int(kept),
        })

    return rows, num_saved


def split_cases(case_dirs, split_file=None, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Return lists of case directories.

    If split_file exists, it must contain case names or case paths under:
    {
      "splits": {
        "train": [...],
        "val": [...],
        "test": [...]
      }
    }
    """
    case_dirs = sorted(case_dirs)
    case_name_to_dir = {os.path.basename(d): d for d in case_dirs}

    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            info = json.load(f).get("splits", {})

        def resolve_cases(items):
            resolved = []
            for item in items:
                # split file may store full path or only case name
                if os.path.isdir(item):
                    resolved.append(item)
                else:
                    case_name = os.path.basename(item)
                    if case_name in case_name_to_dir:
                        resolved.append(case_name_to_dir[case_name])
                    else:
                        print(f"[WARN] Case from split file not found: {item}")
            return resolved

        train_cases = resolve_cases(info.get("train", []))
        val_cases = resolve_cases(info.get("valid", []))
        test_cases = resolve_cases(info.get("test", []))
        return train_cases, val_cases, test_cases

    random.seed(seed)
    shuffled = case_dirs[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = shuffled[:n_train]
    val_cases = shuffled[n_train:n_train + n_val]
    test_cases = shuffled[n_train + n_val:]

    return train_cases, val_cases, test_cases


def check_patient_overlap(splits):
    split_sets = {k: set(os.path.basename(x) for x in v) for k, v in splits.items()}
    names = list(split_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = split_sets[a].intersection(split_sets[b])
            if overlap:
                raise RuntimeError(f"Patient leakage detected between {a} and {b}: {sorted(overlap)[:10]}")


def check_test_continuity(metadata_rows):
    """
    Test split must keep all slices 0..D-1 for each case.
    """
    test_rows = [r for r in metadata_rows if r["split"] == "test"]
    by_case = {}
    for r in test_rows:
        by_case.setdefault(r["case_id"], []).append(r)

    for case_id, rows in by_case.items():
        depth = int(rows[0]["depth"])
        kept_indices = sorted(int(r["slice_index"]) for r in rows if int(r["kept"]) == 1)
        expected = list(range(depth))
        if kept_indices != expected:
            missing = sorted(set(expected) - set(kept_indices))
            raise RuntimeError(
                f"Test case {case_id} is not contiguous. Missing slices: {missing[:20]}"
            )


def write_case_list(path, cases):
    with open(path, "w") as f:
        for case_dir in cases:
            f.write(os.path.basename(case_dir) + "\n")


def prepare_output_dirs(out_root, clean=False):
    if clean and os.path.exists(out_root):
        shutil.rmtree(out_root)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "annotations", split), exist_ok=True)

    os.makedirs(os.path.join(out_root, "metadata"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", default="data/BraTS20/MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--out-root", default="data/medical_seg/brain")
    parser.add_argument("--split-file", default="data/brain_info.json")
    parser.add_argument("--seed", type=int, default=42)

    # Keep previous behavior for train/val unless user explicitly changes it.
    # Test always keeps all slices regardless of this flag.
    parser.add_argument("--keep-empty", action="store_true",
                        help="Keep all empty slices for train/val. Test always keeps all slices.")
    parser.add_argument("--empty-keep-prob", type=float, default=0.15,
                        help="Probability of keeping empty slices for train/val when --keep-empty is not set.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove out-root before preprocessing.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    case_dirs = [
        d for d in glob.glob(os.path.join(args.brats_root, "*"))
        if os.path.isdir(d)
    ]

    train_cases, val_cases, test_cases = split_cases(
        case_dirs,
        split_file=args.split_file,
        seed=args.seed,
    )

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }

    check_patient_overlap(splits)
    prepare_output_dirs(args.out_root, clean=args.clean)

    metadata_rows = []

    for split, cases in splits.items():
        img_dir = os.path.join(args.out_root, "images", split)
        mask_dir = os.path.join(args.out_root, "annotations", split)

        total = 0
        for case_dir in tqdm(cases, desc=f"Processing {split}"):
            rows, saved = save_case_slices(
                case_dir=case_dir,
                out_img_dir=img_dir,
                out_mask_dir=mask_dir,
                split=split,
                keep_empty_all=args.keep_empty,
                empty_keep_prob=args.empty_keep_prob,
            )
            metadata_rows.extend(rows)
            total += saved

        print(f"{split}: {len(cases)} cases, {total} slices saved")

    check_test_continuity(metadata_rows)

    meta_dir = os.path.join(args.out_root, "metadata")
    write_case_list(os.path.join(meta_dir, "train_cases.txt"), train_cases)
    write_case_list(os.path.join(meta_dir, "val_cases.txt"), val_cases)
    write_case_list(os.path.join(meta_dir, "test_cases.txt"), test_cases)

    metadata_csv = os.path.join(meta_dir, "slice_metadata.csv")
    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False)
    print(f"Saved metadata: {metadata_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
