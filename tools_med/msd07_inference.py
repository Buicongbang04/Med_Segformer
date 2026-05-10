#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil

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

DEFAULT_HU_MIN = -125.0
DEFAULT_HU_MAX = 275.0


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
        r"^(volume-|ct-|CT-|case-|Case-|image-|img-)",
        "",
        name
    )

    if name.startswith("Pancreas_"):
        return name

    return "Pancreas_{}".format(name)


def normalize_ct_medsam(
    volume,
    hu_min=-125.0,
    hu_max=275.0,
):
    """
    Pancreas CT preprocessing.

    Same logic as training preprocess:
        - clip HU
        - scale to [0, 1]

    No ImageNet normalization.
    No slice-wise min-max.
    """

    if hu_max <= hu_min:
        raise ValueError(
            "hu_max must be greater than hu_min, got {}, {}".format(
                hu_min,
                hu_max
            )
        )

    volume = volume.astype(np.float32)

    volume = np.clip(
        volume,
        hu_min,
        hu_max
    )

    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)


def prepare_slice_pngs(
    volume_norm,
    case_id,
    image_dir,
):
    """
    Convert normalized 3D CT volume into 2D PNG slices.

    Output:
        image_dir/{case_id}_z000.png
        image_dir/{case_id}_z001.png
        ...

    Each slice is grayscale stacked into 3 channels because SegFormer
    expects RGB-like input.
    """

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    depth = volume_norm.shape[2]

    image_paths = []

    for z in range(depth):
        img_slice = volume_norm[:, :, z]

        img_uint8 = np.clip(
            img_slice * 255.0,
            0,
            255
        ).astype(np.uint8)

        img_3ch = np.stack(
            [img_uint8, img_uint8, img_uint8],
            axis=-1
        )

        out_name = "{}_z{:03d}{}".format(
            case_id,
            z,
            IMAGE_EXT
        )

        out_path = os.path.join(
            image_dir,
            out_name
        )

        cv2.imwrite(
            out_path,
            img_3ch
        )

        image_paths.append(out_path)

    return image_paths


def build_model(
    config_path,
    checkpoint_path,
    device="cuda:0",
):
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
        model.CLASSES = ("background", "target")

    if "meta" in ckpt and "PALETTE" in ckpt["meta"]:
        model.PALETTE = ckpt["meta"]["PALETTE"]
    else:
        model.PALETTE = [[0, 0, 0], [255, 255, 255]]

    # Important: inference_segmentor needs model.cfg
    model.cfg = cfg

    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()

    return model


def normalize_pred_array(pred):
    """
    MMSeg output can be:
        [H, W]
        list([H, W])
        [1, H, W]
        [H, W, 1]

    Return:
        2D class-index mask.
    """

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred = np.asarray(pred)

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[:, :, 0]

    if pred.ndim != 2:
        raise RuntimeError(
            "Prediction must be 2D, got shape={}".format(
                pred.shape
            )
        )

    return pred


def run_slice_inference(
    model,
    image_paths,
    pred_dir,
):
    """
    Run 2D SegFormer inference slice-by-slice.

    Binary segmentation:
        pred == 0: background / non-target
        pred > 0 : target

    Saved PNG:
        0   = background
        255 = target visualization

    Returned volume:
        0 = background
        1 = target
    """

    os.makedirs(
        pred_dir,
        exist_ok=True
    )

    pred_slices = []

    for img_path in image_paths:
        pred = inference_segmentor(
            model,
            img_path
        )

        pred = normalize_pred_array(pred)

        mask01 = (pred > 0).astype(np.uint8)

        mask255 = mask01 * 255

        out_name = os.path.splitext(
            os.path.basename(img_path)
        )[0] + ".png"

        out_path = os.path.join(
            pred_dir,
            out_name
        )

        Image.fromarray(mask255).save(out_path)

        pred_slices.append(mask01)

    pred_volume = np.stack(
        pred_slices,
        axis=2
    ).astype(np.uint8)

    return pred_volume


def compute_volume_metrics(
    pred_mask,
    nifti_img,
):
    """
    Compute predicted target volume using original NIfTI spacing.
    """

    zooms = nifti_img.header.get_zooms()[:3]

    voxel_volume_mm3 = float(
        np.prod(zooms)
    )

    pred_voxels = int(
        (pred_mask > 0).sum()
    )

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


def save_nifti_mask(
    pred_mask,
    ref_img,
    out_path,
):
    """
    Save prediction mask as NIfTI using original affine and header.

    Output mask:
        uint8
        0 = background / non-target
        1 = target
    """

    header = ref_img.header.copy()

    header.set_data_dtype(np.uint8)

    mask_img = nib.Nifti1Image(
        pred_mask.astype(np.uint8),
        affine=ref_img.affine,
        header=header
    )

    nib.save(
        mask_img,
        out_path
    )


def save_metadata_csv(
    metrics,
    out_path,
):
    df = pd.DataFrame([metrics])

    df.to_csv(
        out_path,
        index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clinical-style end-to-end inference pipeline for pancreas CT "
            "segmentation using a 2D SegFormer model."
        )
    )

    parser.add_argument(
        "--nifti",
        required=True,
        help="Input CT NIfTI file: .nii or .nii.gz"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="MMSeg SegFormer config file."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained checkpoint .pth file."
    )

    parser.add_argument(
        "--out-root",
        default="work_dirs/pancreas_ct/infer",
        help="Output root directory."
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu."
    )

    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional case ID. If not provided, inferred from NIfTI filename."
    )

    parser.add_argument(
        "--target-name",
        default="pancreas",
        help="Only used for metadata/logging, e.g. pancreas or pancreas_tumor."
    )

    parser.add_argument(
        "--hu-min",
        type=float,
        default=DEFAULT_HU_MIN,
        help="Minimum HU for CT windowing."
    )

    parser.add_argument(
        "--hu-max",
        type=float,
        default=DEFAULT_HU_MAX,
        help="Maximum HU for CT windowing."
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing case output folder before inference."
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

    os.makedirs(
        case_out_dir,
        exist_ok=True
    )

    print("[INFO] Loading pancreas CT NIfTI...")
    print(args.nifti)

    nifti_img = nib.load(args.nifti)

    ct = nifti_img.get_fdata(dtype=np.float32)

    if ct.ndim != 3:
        raise RuntimeError(
            "Expected 3D CT volume, got shape={}".format(
                ct.shape
            )
        )

    print("[INFO] Input shape:")
    print(ct.shape)

    print("[INFO] CT spacing:")
    print(nifti_img.header.get_zooms()[:3])

    print("[INFO] Normalizing CT with HU window:")
    print("[{}, {}]".format(args.hu_min, args.hu_max))

    ct_norm = normalize_ct_medsam(
        ct,
        hu_min=args.hu_min,
        hu_max=args.hu_max
    )

    print("[INFO] Exporting 2D PNG slices...")

    image_paths = prepare_slice_pngs(
        ct_norm,
        case_id,
        image_dir
    )

    print("[INFO] Number of slices:")
    print(len(image_paths))

    print("[INFO] Building SegFormer model...")

    model = build_model(
        args.config,
        args.checkpoint,
        args.device
    )

    print("[INFO] Running 2D slice inference...")

    pred_mask = run_slice_inference(
        model,
        image_paths,
        pred_dir
    )

    if pred_mask.shape != ct.shape:
        raise RuntimeError(
            "Prediction shape mismatch: pred={} input={}".format(
                pred_mask.shape,
                ct.shape
            )
        )

    pred_nifti_path = os.path.join(
        case_out_dir,
        "{}_pred_mask.nii.gz".format(case_id)
    )

    print("[INFO] Saving 3D prediction NIfTI...")

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
    metrics["input_nifti"] = args.nifti
    metrics["num_slices"] = int(ct.shape[2])
    metrics["height"] = int(ct.shape[0])
    metrics["width"] = int(ct.shape[1])
    metrics["hu_min"] = float(args.hu_min)
    metrics["hu_max"] = float(args.hu_max)
    metrics["target_name"] = args.target_name
    metrics["pred_mask_nifti"] = pred_nifti_path
    metrics["pred_mask_png_dir"] = pred_dir
    metrics["preprocessed_png_dir"] = image_dir

    metadata_csv = os.path.join(
        case_out_dir,
        "inference_metadata.csv"
    )

    save_metadata_csv(
        metrics,
        metadata_csv
    )

    print("\n========== PANCREAS CT CLINICAL INFERENCE SUMMARY ==========")
    print("Case ID:", case_id)
    print("Target:", args.target_name)
    print("Input shape:", ct.shape)
    print("Pred mask shape:", pred_mask.shape)
    print("Pred voxels:", metrics["pred_voxels"])

    print("Voxel size:")
    print(
        "x={:.6f} mm, y={:.6f} mm, z={:.6f} mm".format(
            metrics["voxel_size_x_mm"],
            metrics["voxel_size_y_mm"],
            metrics["voxel_size_z_mm"]
        )
    )

    print("Voxel volume:")
    print(
        "{:.6f} mm3".format(
            metrics["voxel_volume_mm3"]
        )
    )

    print("Pred target volume:")
    print(
        "{:.2f} mm3".format(
            metrics["pred_volume_mm3"]
        )
    )

    print("Pred target volume:")
    print(
        "{:.4f} mL".format(
            metrics["pred_volume_ml"]
        )
    )

    print("Saved pred NIfTI:")
    print(pred_nifti_path)

    print("Saved pred PNG dir:")
    print(pred_dir)

    print("Saved metadata:")
    print(metadata_csv)


if __name__ == "__main__":
    main()