import json
import os
import glob
import argparse
import random
import shutil
import re

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXT = ".png"

DEFAULT_LIVER_WINDOW = (-200.0, 250.0)

# Ablation option:
# ABLATION_LIVER_WINDOW = (-150.0, 250.0)

LITS_TUMOR_LABEL_THRESHOLD = 1


def normalize_ct(volume, hu_min=-200.0, hu_max=250.0):
    if hu_max <= hu_min:
        raise ValueError(f"hu_max must be greater than hu_min, got {hu_min}, {hu_max}")

    volume = volume.astype(np.float32)
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def get_lits_case_id(path):
    base = os.path.basename(path)
    base = base.replace(".nii.gz", "").replace(".nii", "")
    match = re.search(r"(\d+)$", base)
    if match is None:
        return base
    return match.group(1)


def find_lits_pairs(lits_root):
    image_patterns = [
        os.path.join(lits_root, "volume-*.nii"),
        os.path.join(lits_root, "volume-*.nii.gz"),
        os.path.join(lits_root, "**", "volume-*.nii"),
        os.path.join(lits_root, "**", "volume-*.nii.gz"),
    ]
    label_patterns = [
        os.path.join(lits_root, "segmentation-*.nii"),
        os.path.join(lits_root, "segmentation-*.nii.gz"),
        os.path.join(lits_root, "**", "segmentation-*.nii"),
        os.path.join(lits_root, "**", "segmentation-*.nii.gz"),
    ]

    image_paths = []
    label_paths = []

    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern, recursive=True))
    for pattern in label_patterns:
        label_paths.extend(glob.glob(pattern, recursive=True))

    image_paths = sorted(set(image_paths))
    label_paths = sorted(set(label_paths))

    image_by_id = {get_lits_case_id(p): p for p in image_paths}
    label_by_id = {get_lits_case_id(p): p for p in label_paths}

    common_ids = sorted(
        set(image_by_id.keys()).intersection(label_by_id.keys()),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    missing_labels = sorted(set(image_by_id.keys()) - set(label_by_id.keys()))
    missing_images = sorted(set(label_by_id.keys()) - set(image_by_id.keys()))

    if missing_labels:
        print(f"[WARN] Missing labels for image case ids: {missing_labels[:20]}")
    if missing_images:
        print(f"[WARN] Missing images for label case ids: {missing_images[:20]}")

    return [
        {
            "case_id": case_id,
            "image_path": image_by_id[case_id],
            "label_path": label_by_id[case_id],
        }
        for case_id in common_ids
    ]


def should_keep_slice(split, has_tumor, keep_empty_all, empty_keep_prob):
    if split == "test":
        return True

    if keep_empty_all:
        return True

    if has_tumor:
        return True

    return random.random() < empty_keep_prob


def save_case_slices(
    case_info,
    out_img_dir,
    out_mask_dir,
    split,
    hu_min=-200.0,
    hu_max=250.0,
    keep_empty_all=False,
    empty_keep_prob=0.15,
):
    case_id = str(case_info["case_id"])
    case_name = f"LiTS_{case_id}"

    image_path = case_info["image_path"]
    label_path = case_info["label_path"]

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)

    if image.shape != label.shape:
        print(f"[SKIP] {case_name}: CT/label shape mismatch {image.shape} vs {label.shape}")
        return [], 0

    image = normalize_ct(image, hu_min=hu_min, hu_max=hu_max)

    # LiTS17:
    # 0 = background
    # 1 = liver
    # 2 = tumor
    #
    # Tumor-only binary target:
    # 0 = non-tumor
    # 1 = tumor
    mask = (label > LITS_TUMOR_LABEL_THRESHOLD).astype(np.uint8)

    rows = []
    num_saved = 0
    depth = image.shape[2]

    for z in range(depth):
        img_slice = image[:, :, z]
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

            # MMSeg CustomDataset expects class indices, not 0/255 masks.
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
            "source_image_path": image_path,
            "source_label_path": label_path,
            "slice_index": z,
            "depth": depth,
            "image_path": img_path,
            "mask_path": mask_path,
            "has_tumor": int(has_tumor),
            "tumor_voxels": tumor_voxels,
            "kept": int(kept),
            "hu_min": float(hu_min),
            "hu_max": float(hu_max),
            "target": "liver_tumor_only",
        })

    return rows, num_saved


def split_cases(case_infos, split_file=None, train_ratio=0.8, val_ratio=0.1, seed=42):
    case_infos = sorted(
        case_infos,
        key=lambda x: int(x["case_id"]) if str(x["case_id"]).isdigit() else str(x["case_id"]),
    )

    case_by_id = {str(x["case_id"]): x for x in case_infos}
    case_by_name = {f"LiTS_{x['case_id']}": x for x in case_infos}

    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            info = json.load(f).get("splits", {})

        def resolve_cases(items):
            resolved = []
            for item in items:
                raw = str(item)
                base = os.path.basename(raw)
                case_id = get_lits_case_id(base)

                if raw in case_by_id:
                    resolved.append(case_by_id[raw])
                elif base in case_by_name:
                    resolved.append(case_by_name[base])
                elif case_id in case_by_id:
                    resolved.append(case_by_id[case_id])
                else:
                    print(f"[WARN] Case from split file not found: {item}")

            return resolved

        train_cases = resolve_cases(info.get("train", []))
        val_cases = resolve_cases(info.get("val", info.get("valid", [])))
        test_cases = resolve_cases(info.get("test", []))

        return train_cases, val_cases, test_cases

    random.seed(seed)
    shuffled = case_infos[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = shuffled[:n_train]
    val_cases = shuffled[n_train:n_train + n_val]
    test_cases = shuffled[n_train + n_val:]

    return train_cases, val_cases, test_cases


def check_patient_overlap(splits):
    split_sets = {k: set(str(x["case_id"]) for x in v) for k, v in splits.items()}

    names = list(split_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = split_sets[a].intersection(split_sets[b])
            if overlap:
                raise RuntimeError(
                    f"Patient leakage detected between {a} and {b}: {sorted(overlap)[:10]}"
                )


def check_test_continuity(metadata_rows):
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
        for case_info in cases:
            f.write(str(case_info["case_id"]) + "\n")


def prepare_output_dirs(out_root, clean=False):
    if clean and os.path.exists(out_root):
        shutil.rmtree(out_root)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "annotations", split), exist_ok=True)

    os.makedirs(os.path.join(out_root, "metadata"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lits-root",
        default="data/LITS17",
        help="Root folder containing LiTS17 volume-*.nii and segmentation-*.nii files.",
    )
    parser.add_argument(
        "--out-root",
        default="data/medical_seg/liver",
        help="Output root for 2D PNG images, annotations, and metadata.",
    )
    parser.add_argument(
        "--split-file",
        default="data/liver_info.json",
        help="Optional JSON split file. Leave empty to random split.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Main HU window:
    # [-200, 250]
    parser.add_argument("--hu-min", type=float, default=DEFAULT_LIVER_WINDOW[0])
    parser.add_argument("--hu-max", type=float, default=DEFAULT_LIVER_WINDOW[1])

    # Ablation HU window:
    # To test narrower window, run:
    # --hu-min -150 --hu-max 250

    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep all empty slices for train/val. Test always keeps all slices.",
    )
    parser.add_argument(
        "--empty-keep-prob",
        type=float,
        default=0.15,
        help="Probability of keeping empty slices for train/val when --keep-empty is not set.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove out-root before preprocessing.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    case_infos = find_lits_pairs(args.lits_root)

    if len(case_infos) == 0:
        raise RuntimeError(
            f"No LiTS17 image/label pairs found under {args.lits_root}. "
            "Expected files like volume-0.nii(.gz) and segmentation-0.nii(.gz)."
        )

    print(f"Found {len(case_infos)} LiTS17 cases")
    print(f"Using CT HU window: [{args.hu_min}, {args.hu_max}]")
    print("Target mask: tumor-only binary mask, original label 2 -> class 1")

    train_cases, val_cases, test_cases = split_cases(
        case_infos,
        split_file=args.split_file if args.split_file else None,
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

        for case_info in tqdm(cases, desc=f"Processing {split}"):
            rows, saved = save_case_slices(
                case_info=case_info,
                out_img_dir=img_dir,
                out_mask_dir=mask_dir,
                split=split,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
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