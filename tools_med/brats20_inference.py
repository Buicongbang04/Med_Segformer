#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
from typing import List, Dict

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor


IMAGE_EXT = ".png"


def strip_nii_suffix(path):
    name = os.path.basename(path)

    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]

    return name


def infer_case_id(nifti_path):
    name = strip_nii_suffix(nifti_path)

    name = re.sub(
        r"_(flair|t1|t1ce|t2)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    return name


def normalize_brats_flair(volume):
    """
    BraTS MRI FLAIR preprocessing.

    - foreground-only percentile clipping
    - foreground-only z-score
    - clip [-5, 5]
    - rescale [0,1]
    """

    volume = volume.astype(np.float32)

    foreground = volume > 0

    if foreground.sum() == 0:
        return np.zeros_like(volume, dtype=np.float32)

    fg_values = volume[foreground]

    p1, p99 = np.percentile(fg_values, [1, 99])

    volume_clipped = volume.copy()

    volume_clipped[foreground] = np.clip(
        volume_clipped[foreground],
        p1,
        p99
    )

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


def prepare_slice_pngs(volume_norm, case_id, image_dir):
    """
    Convert normalized 3D volume into 2D PNG slices.
    """

    os.makedirs(image_dir, exist_ok=True)

    depth = volume_norm.shape[2]

    image_paths = []

    for z in range(depth):

        img_slice = volume_norm[:, :, z]

        img_uint8 = np.clip(
            img_slice * 255.0,
            0,
            255
        ).astype(np.uint8)

        # 3-channel grayscale
        img_3ch = np.stack(
            [img_uint8, img_uint8, img_uint8],
            axis=-1
        )

        out_name = "{}_z{:03d}{}".format(
            case_id,
            z,
            IMAGE_EXT
        )

        out_path = os.path.join(image_dir, out_name)

        cv2.imwrite(out_path, img_3ch)

        image_paths.append(out_path)

    return image_paths


def build_model(config_path, checkpoint_path, device="cuda:0"):

    cfg = Config.fromfile(config_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_segmentor(
        cfg.model,
        test_cfg=cfg.get("test_cfg")
    )

    ckpt = load_checkpoint(
        model,
        checkpoint_path,
        map_location="cpu"
    )

    if "meta" in ckpt and "CLASSES" in ckpt["meta"]:
        model.CLASSES = ckpt["meta"]["CLASSES"]
    else:
        model.CLASSES = ("background", "tumor")

    if "meta" in ckpt and "PALETTE" in ckpt["meta"]:
        model.PALETTE = ckpt["meta"]["PALETTE"]
    else:
        model.PALETTE = [[0, 0, 0], [255, 255, 255]]

    # Quan trọng: inference_segmentor cần model.cfg
    model.cfg = cfg

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()

    return model

def normalize_pred_array(pred):

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred = np.asarray(pred)

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:, :, 0]

    if pred.ndim != 2:
        raise RuntimeError(
            "Prediction must be 2D, got shape={}".format(pred.shape)
        )

    return pred


def run_slice_inference(model, image_paths, pred_dir):

    os.makedirs(pred_dir, exist_ok=True)

    pred_slices = []

    for img_path in image_paths:

        pred = inference_segmentor(model, img_path)

        pred = normalize_pred_array(pred)

        mask01 = (pred > 0).astype(np.uint8)

        mask255 = mask01 * 255

        out_name = os.path.splitext(
            os.path.basename(img_path)
        )[0] + ".png"

        out_path = os.path.join(pred_dir, out_name)

        Image.fromarray(mask255).save(out_path)

        pred_slices.append(mask01)

    pred_volume = np.stack(
        pred_slices,
        axis=2
    ).astype(np.uint8)

    return pred_volume


def compute_volume_metrics(pred_mask, nifti_img):

    zooms = nifti_img.header.get_zooms()[:3]

    voxel_volume_mm3 = float(np.prod(zooms))

    pred_voxels = int((pred_mask > 0).sum())

    pred_volume_mm3 = float(
        pred_voxels * voxel_volume_mm3
    )

    pred_volume_ml = float(
        pred_volume_mm3 / 1000.0
    )

    return {
        "pred_voxels": pred_voxels,
        "voxel_size_x_mm": float(zooms[0]),
        "voxel_size_y_mm": float(zooms[1]),
        "voxel_size_z_mm": float(zooms[2]),
        "voxel_volume_mm3": voxel_volume_mm3,
        "pred_volume_mm3": pred_volume_mm3,
        "pred_volume_ml": pred_volume_ml,
    }


def save_nifti_mask(pred_mask, ref_img, out_path):

    header = ref_img.header.copy()

    header.set_data_dtype(np.uint8)

    mask_img = nib.Nifti1Image(
        pred_mask.astype(np.uint8),
        affine=ref_img.affine,
        header=header
    )

    nib.save(mask_img, out_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--out-root",
        default="work_dirs/brats20/infer"
    )

    parser.add_argument(
        "--device",
        default="cuda:0"
    )

    parser.add_argument(
        "--case-id",
        default=None
    )

    parser.add_argument(
        "--clean",
        action="store_true"
    )

    args = parser.parse_args()

    if not os.path.exists(args.nifti):
        raise FileNotFoundError(args.nifti)

    case_id = args.case_id

    if case_id is None:
        case_id = infer_case_id(args.nifti)

    case_out_dir = os.path.join(
        args.out_root,
        case_id
    )

    image_dir = os.path.join(
        case_out_dir,
        "images"
    )

    pred_dir = os.path.join(
        case_out_dir,
        "preds_mask"
    )

    if args.clean and os.path.exists(case_out_dir):
        shutil.rmtree(case_out_dir)

    os.makedirs(case_out_dir, exist_ok=True)

    print("[INFO] Loading NIfTI...")
    print(args.nifti)

    nifti_img = nib.load(args.nifti)

    flair = nifti_img.get_fdata()

    if flair.ndim != 3:
        raise RuntimeError(
            "Expected 3D volume, got shape={}".format(
                flair.shape
            )
        )

    print("[INFO] Input shape:")
    print(flair.shape)

    print("[INFO] Normalizing FLAIR...")

    flair_norm = normalize_brats_flair(flair)

    print("[INFO] Exporting PNG slices...")

    image_paths = prepare_slice_pngs(
        flair_norm,
        case_id,
        image_dir
    )

    print("[INFO] Building model...")

    model = build_model(
        args.config,
        args.checkpoint,
        args.device
    )

    print("[INFO] Running inference...")

    pred_mask = run_slice_inference(
        model,
        image_paths,
        pred_dir
    )

    if pred_mask.shape != flair.shape:
        raise RuntimeError(
            "Prediction shape mismatch: pred={} input={}".format(
                pred_mask.shape,
                flair.shape
            )
        )

    pred_nifti_path = os.path.join(
        case_out_dir,
        "{}_pred_mask.nii.gz".format(case_id)
    )

    save_nifti_mask(
        pred_mask,
        nifti_img,
        pred_nifti_path
    )

    metrics = compute_volume_metrics(
        pred_mask,
        nifti_img
    )

    metrics["case_id"] = case_id
    metrics["num_slices"] = int(flair.shape[2])
    metrics["height"] = int(flair.shape[0])
    metrics["width"] = int(flair.shape[1])
    metrics["pred_mask_nifti"] = pred_nifti_path
    metrics["pred_mask_png_dir"] = pred_dir
    metrics["preprocessed_png_dir"] = image_dir

    print("\n========== INFERENCE SUMMARY ==========")

    print("Case ID:", end=" ")
    print(case_id)

    print("Pred voxels:", end=" ")
    print(metrics["pred_voxels"])

    print("Voxel volume:", end=" ")
    print("{:.6f} mm3".format(
        metrics["voxel_volume_mm3"]
    ))

    print("Pred volume:", end=" ")
    print("{:.2f} mm3".format(
        metrics["pred_volume_mm3"]
    ))

    print("Pred volume:", end=" ")
    print("{:.4f} mL".format(
        metrics["pred_volume_ml"]
    ))


if __name__ == "__main__":
    main()