#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

DEFAULT_PANCREAS_WINDOW = (-125.0, 275.0)

# Ablation options:
# ABLATION_PANCREAS_WINDOW_1 = (-100.0, 240.0)
# ABLATION_PANCREAS_WINDOW_2 = (-150.0, 250.0)


def normalize_ct_medsam(volume, hu_min=-125.0, hu_max=275.0):
    """
    CT normalization following MedSAM-style windowing:
        1. clip HU
        2. scale to [0, 1]

    Important:
        This is done on the CT volume intensity directly.
        No ImageNet normalization.
        No per-slice min-max.
        No histogram equalization.
    """
    if hu_max <= hu_min:
        raise ValueError(
            "hu_max must be greater than hu_min, got {}, {}".format(
                hu_min,
                hu_max
            )
        )

    volume = volume.astype(np.float32)
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)


def strip_nii_suffix(path):
    base = os.path.basename(path)

    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    return base


def get_ct_case_id(path):
    """
    Extract case ID from common pancreas CT filenames.

    Supported examples:
        pancreas_001.nii.gz
        case_001.nii.gz
        volume-001.nii.gz
        PANCREAS_001.nii.gz
    """
    base = strip_nii_suffix(path)

    match = re.search(r"(\d+)$", base)

    if match is None:
        return base

    return match.group(1)


def find_ct_pairs(
    ct_root,
    image_prefixes=None,
    label_prefixes=None,
):
    """
    Find CT image/label NIfTI pairs.

    Default patterns are intentionally broad to support:
        imagesTr/case_001.nii.gz
        labelsTr/case_001.nii.gz
        volume-001.nii.gz / segmentation-001.nii.gz
        pancreas_001.nii.gz / label_001.nii.gz
    """

    if image_prefixes is None:
        image_prefixes = [
            "volume-*",
            "case_*",
            "case-*",
            "pancreas_*",
            "PANCREAS_*",
            "img_*",
            "image_*",
        ]

    if label_prefixes is None:
        label_prefixes = [
            "segmentation-*",
            "label_*",
            "labels_*",
            "mask_*",
            "case_*",
            "case-*",
            "pancreas_*",
            "PANCREAS_*",
        ]

    image_paths = []
    label_paths = []

    image_dirs = [
        ct_root,
        os.path.join(ct_root, "imagesTr"),
        os.path.join(ct_root, "images"),
    ]

    label_dirs = [
        ct_root,
        os.path.join(ct_root, "labelsTr"),
        os.path.join(ct_root, "labels"),
        os.path.join(ct_root, "annotations"),
    ]

    for image_dir in image_dirs:
        for prefix in image_prefixes:
            image_paths.extend(
                glob.glob(
                    os.path.join(image_dir, prefix + ".nii"),
                    recursive=True
                )
            )
            image_paths.extend(
                glob.glob(
                    os.path.join(image_dir, prefix + ".nii.gz"),
                    recursive=True
                )
            )

    for label_dir in label_dirs:
        for prefix in label_prefixes:
            label_paths.extend(
                glob.glob(
                    os.path.join(label_dir, prefix + ".nii"),
                    recursive=True
                )
            )
            label_paths.extend(
                glob.glob(
                    os.path.join(label_dir, prefix + ".nii.gz"),
                    recursive=True
                )
            )

    image_paths = sorted(set(image_paths))
    label_paths = sorted(set(label_paths))

    image_by_id = {
        get_ct_case_id(p): p
        for p in image_paths
    }

    label_by_id = {
        get_ct_case_id(p): p
        for p in label_paths
    }

    common_ids = sorted(
        set(image_by_id.keys()).intersection(label_by_id.keys()),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    missing_labels = sorted(
        set(image_by_id.keys()) - set(label_by_id.keys())
    )

    missing_images = sorted(
        set(label_by_id.keys()) - set(image_by_id.keys())
    )

    if missing_labels:
        print(
            "[WARN] Missing labels for image case ids:",
            missing_labels[:20]
        )

    if missing_images:
        print(
            "[WARN] Missing images for label case ids:",
            missing_images[:20]
        )

    return [
        {
            "case_id": case_id,
            "image_path": image_by_id[case_id],
            "label_path": label_by_id[case_id],
        }
        for case_id in common_ids
    ]


def build_binary_mask(label, target_mode):
    """
    Build binary target mask for pancreas CT.

    target_mode:
        pancreas_only:
            for datasets with:
                0 = background
                1 = pancreas
            or:
                0 = background
                1 = pancreas
                2 = tumor

            mask = label > 0

        pancreas_tumor_only:
            for datasets with:
                0 = background
                1 = pancreas
                2 = tumor

            mask = label == 2
    """

    if target_mode == "pancreas_only":
        return (label > 0).astype(np.uint8)

    if target_mode == "pancreas_tumor_only":
        return (label == 2).astype(np.uint8)

    raise ValueError(
        "Unknown target_mode: {}. Use pancreas_only or pancreas_tumor_only.".format(
            target_mode
        )
    )


def should_keep_slice(split, has_target, keep_empty_all, empty_keep_prob):
    if split == "test":
        return True

    if keep_empty_all:
        return True

    if has_target:
        return True

    return random.random() < empty_keep_prob


def save_case_slices(
    case_info,
    out_img_dir,
    out_mask_dir,
    split,
    target_mode,
    hu_min=-125.0,
    hu_max=275.0,
    keep_empty_all=False,
    empty_keep_prob=0.4,
):
    case_id = str(case_info["case_id"])
    case_name = "Pancreas_{}".format(case_id)

    image_path = case_info["image_path"]
    label_path = case_info["label_path"]

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)

    if image.shape != label.shape:
        print(
            "[SKIP] {}: CT/label shape mismatch {} vs {}".format(
                case_name,
                image.shape,
                label.shape
            )
        )
        return [], 0

    image = normalize_ct_medsam(
        image,
        hu_min=hu_min,
        hu_max=hu_max
    )

    mask = build_binary_mask(
        label,
        target_mode=target_mode
    )

    rows = []
    num_saved = 0
    depth = image.shape[2]

    for z in range(depth):
        img_slice = image[:, :, z]
        mask_slice = mask[:, :, z]

        target_voxels = int(mask_slice.sum())
        has_target = target_voxels > 0

        kept = should_keep_slice(
            split=split,
            has_target=has_target,
            keep_empty_all=keep_empty_all,
            empty_keep_prob=empty_keep_prob,
        )

        filename = "{}_z{:03d}{}".format(
            case_name,
            z,
            IMAGE_EXT
        )

        img_path = os.path.join(
            out_img_dir,
            filename
        )

        mask_path = os.path.join(
            out_mask_dir,
            filename
        )

        if kept:
            img_uint8 = np.clip(
                img_slice * 255.0,
                0,
                255
            ).astype(np.uint8)

            img_3ch = np.stack(
                [img_uint8, img_uint8, img_uint8],
                axis=-1
            )

            # MMSeg CustomDataset expects class-index masks:
            # 0 = background
            # 1 = target
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
            "has_target": int(has_target),
            "target_voxels": target_voxels,
            "kept": int(kept),
            "hu_min": float(hu_min),
            "hu_max": float(hu_max),
            "target_mode": target_mode,
        })

    return rows, num_saved


def split_cases(
    case_infos,
    split_file=None,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
):
    case_infos = sorted(
        case_infos,
        key=lambda x: int(x["case_id"]) if str(x["case_id"]).isdigit() else str(x["case_id"]),
    )

    case_by_id = {
        str(x["case_id"]): x
        for x in case_infos
    }

    case_by_name = {
        "Pancreas_{}".format(x["case_id"]): x
        for x in case_infos
    }

    if split_file and os.path.exists(split_file):
        with open(split_file, "r") as f:
            info = json.load(f).get("splits", {})

        def resolve_cases(items):
            resolved = []

            for item in items:
                raw = str(item)
                base = os.path.basename(raw)
                case_id = get_ct_case_id(base)

                if raw in case_by_id:
                    resolved.append(case_by_id[raw])
                elif base in case_by_name:
                    resolved.append(case_by_name[base])
                elif case_id in case_by_id:
                    resolved.append(case_by_id[case_id])
                else:
                    print(
                        "[WARN] Case from split file not found:",
                        item
                    )

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
    split_sets = {
        k: set(str(x["case_id"]) for x in v)
        for k, v in splits.items()
    }

    names = list(split_sets.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            overlap = split_sets[a].intersection(split_sets[b])

            if overlap:
                raise RuntimeError(
                    "Patient leakage detected between {} and {}: {}".format(
                        a,
                        b,
                        sorted(overlap)[:10]
                    )
                )


def check_test_continuity(metadata_rows):
    test_rows = [
        r for r in metadata_rows
        if r["split"] == "test"
    ]

    by_case = {}

    for r in test_rows:
        by_case.setdefault(
            r["case_id"],
            []
        ).append(r)

    for case_id, rows in by_case.items():
        depth = int(rows[0]["depth"])

        kept_indices = sorted(
            int(r["slice_index"])
            for r in rows
            if int(r["kept"]) == 1
        )

        expected = list(range(depth))

        if kept_indices != expected:
            missing = sorted(
                set(expected) - set(kept_indices)
            )

            raise RuntimeError(
                "Test case {} is not contiguous. Missing slices: {}".format(
                    case_id,
                    missing[:20]
                )
            )


def write_case_list(path, cases):
    with open(path, "w") as f:
        for case_info in cases:
            f.write(str(case_info["case_id"]) + "\n")


def prepare_output_dirs(out_root, clean=False):
    if clean and os.path.exists(out_root):
        shutil.rmtree(out_root)

    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(out_root, "images", split),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(out_root, "annotations", split),
            exist_ok=True
        )

    os.makedirs(
        os.path.join(out_root, "metadata"),
        exist_ok=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MSD07 CT NIfTI dataset into 2D PNGs for SegFormer."
    )

    parser.add_argument(
        "--ct-root",
        default="data/Pancreas",
        help="Root folder containing pancreas CT images and labels."
    )

    parser.add_argument(
        "--out-root",
        default="data/medical_seg/msd",
        help="Output root for 2D PNG images, annotations, and metadata."
    )

    parser.add_argument(
        "--split-file",
        default="data/msd07.json",
        help="Optional JSON split file. Leave empty to random split."
    )

    parser.add_argument(
        "--target-mode",
        default="pancreas_tumor_only",
        choices=["pancreas_only", "pancreas_tumor_only"],
        help=(
            "pancreas_only: mask = label > 0. "
            "pancreas_tumor_only: mask = label == 2."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--hu-min",
        type=float,
        default=DEFAULT_PANCREAS_WINDOW[0]
    )

    parser.add_argument(
        "--hu-max",
        type=float,
        default=DEFAULT_PANCREAS_WINDOW[1]
    )

    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep all empty slices for train/val. Test always keeps all slices."
    )

    parser.add_argument(
        "--empty-keep-prob",
        type=float,
        default=0.4,
        help=(
            "Probability of keeping empty slices for train/val. "
            "Pancreas is small, so default is higher than LiTS17 liver."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove out-root before preprocessing."
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    case_infos = find_ct_pairs(args.ct_root)

    if len(case_infos) == 0:
        raise RuntimeError(
            "No pancreas CT image/label pairs found under {}.".format(
                args.ct_root
            )
        )

    print("Found {} pancreas CT cases".format(len(case_infos)))
    print("Using CT HU window: [{}, {}]".format(args.hu_min, args.hu_max))
    print("Target mode:", args.target_mode)

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

    prepare_output_dirs(
        args.out_root,
        clean=args.clean
    )

    metadata_rows = []

    for split, cases in splits.items():
        img_dir = os.path.join(
            args.out_root,
            "images",
            split
        )

        mask_dir = os.path.join(
            args.out_root,
            "annotations",
            split
        )

        total = 0

        for case_info in tqdm(cases, desc="Processing {}".format(split)):
            rows, saved = save_case_slices(
                case_info=case_info,
                out_img_dir=img_dir,
                out_mask_dir=mask_dir,
                split=split,
                target_mode=args.target_mode,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
                keep_empty_all=args.keep_empty,
                empty_keep_prob=args.empty_keep_prob,
            )

            metadata_rows.extend(rows)
            total += saved

        print(
            "{}: {} cases, {} slices saved".format(
                split,
                len(cases),
                total
            )
        )

    check_test_continuity(metadata_rows)

    meta_dir = os.path.join(
        args.out_root,
        "metadata"
    )

    write_case_list(
        os.path.join(meta_dir, "train_cases.txt"),
        train_cases
    )

    write_case_list(
        os.path.join(meta_dir, "val_cases.txt"),
        val_cases
    )

    write_case_list(
        os.path.join(meta_dir, "test_cases.txt"),
        test_cases
    )

    metadata_csv = os.path.join(
        meta_dir,
        "slice_metadata.csv"
    )

    pd.DataFrame(metadata_rows).to_csv(
        metadata_csv,
        index=False
    )

    print("Saved metadata:", metadata_csv)
    print("Done.")


if __name__ == "__main__":
    main()