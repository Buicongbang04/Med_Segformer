import os
import re
import shutil
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmcv.parallel import MMDataParallel
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_case_and_slice(filename):
    """
    Extract case ID and slice index from filename.

    Supported examples:
        volume-0_slice_000.png
        volume-0_z000.png
        volume-0_000.png
    """
    name = os.path.splitext(os.path.basename(filename))[0]

    if "_slice_" in name:
        case_id, slice_id = name.rsplit("_slice_", 1)
        return case_id, int(slice_id)

    match = re.match(r"(.+)_z(\d+)$", name)
    if match:
        case_id = match.group(1)
        slice_id = int(match.group(2))
        return case_id, slice_id

    nums = re.findall(r"\d+", name)
    if not nums:
        raise ValueError(f"Cannot parse slice index from filename: {filename}")

    slice_id_str = nums[-1]
    slice_id = int(slice_id_str)
    idx = name.rfind(slice_id_str)
    case_id = name[:idx].rstrip("_-")

    return case_id, slice_id


def load_binary_mask(path):
    """
    Load binary PNG mask.

    For LiTS17 tumor-only preprocessing:
        original label:
            0 = background
            1 = liver
            2 = tumor

        preprocessed GT:
            mask = (label > 1).astype(np.uint8)

    Therefore, here:
        arr > 0 means tumor.
    """
    arr = np.array(Image.open(path))

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr > 0


def group_slices(folder):
    """
    Return:
        case_id -> list of (slice_index, file_path)
    """
    cases = defaultdict(list)

    for fname in os.listdir(folder):
        if not fname.lower().endswith(VALID_IMAGE_EXTS):
            continue

        case_id, slice_id = parse_case_and_slice(fname)
        cases[case_id].append((slice_id, os.path.join(folder, fname)))

    return cases


def assert_contiguous(case_id, slice_items):
    """
    Ensure slice indices are continuous.
    """
    indices = sorted(i for i, _ in slice_items)

    if len(indices) == 0:
        raise RuntimeError(f"Case {case_id} has no slices.")

    expected = list(range(indices[0], indices[-1] + 1))

    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise RuntimeError(
            f"Case {case_id} is not contiguous. "
            f"Found range z{indices[0]:03d}-z{indices[-1]:03d}, "
            f"missing {len(missing)} slices, first missing: {missing[:20]}"
        )


def stack_volume(case_id, slice_items, require_contiguous=True):
    """
    Stack 2D binary slices into 3D volume:

        [D, H, W]
    """
    if require_contiguous:
        assert_contiguous(case_id, slice_items)

    slice_items = sorted(slice_items, key=lambda x: x[0])
    volume = [load_binary_mask(path) for _, path in slice_items]

    return np.stack(volume, axis=0)


def dice_score(pred, gt, eps=1e-8):
    """
    3D Dice score.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return float((2.0 * inter) / (denom + eps))


def compute_volume_metrics(pred, gt, voxel_volume_mm3):
    """
    Compute tumor volume metrics.
    """
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())

    pred_mm3 = float(pred_voxels * voxel_volume_mm3)
    gt_mm3 = float(gt_voxels * voxel_volume_mm3)

    abs_error_mm3 = abs(pred_mm3 - gt_mm3)

    if gt_mm3 == 0:
        rel_error = 0.0 if pred_mm3 == 0 else np.inf
    else:
        rel_error = abs_error_mm3 / gt_mm3

    return {
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_volume_mm3": pred_mm3,
        "gt_volume_mm3": gt_mm3,
        "abs_volume_error_mm3": abs_error_mm3,
        "relative_volume_error": rel_error,
    }


def run_mmseg_prediction(config, checkpoint, pred_dir, clean_pred_dir=True):
    """
    Run MMSeg inference like tools/test.py.

    Save binary tumor predictions to:

        pred_dir/preds_mask/*.png

    Saved PNG:
        0   = background
        255 = tumor visualization

    Logical mask:
        0 = non-tumor
        1 = tumor
    """
    if clean_pred_dir and os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    os.makedirs(pred_dir, exist_ok=True)

    mask_dir = os.path.join(pred_dir, "preds_mask")
    os.makedirs(mask_dir, exist_ok=True)

    cfg = Config.fromfile(config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(
        cfg.model,
        test_cfg=cfg.get("test_cfg")
    )

    ckpt = load_checkpoint(model, checkpoint, map_location="cpu")

    if "meta" in ckpt and "CLASSES" in ckpt["meta"]:
        model.CLASSES = ckpt["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if "meta" in ckpt and "PALETTE" in ckpt["meta"]:
        model.PALETTE = ckpt["meta"]["PALETTE"]
    else:
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(
        model,
        data_loader,
        show=False,
        out_dir=None,
        efficient_test=False,
    )

    if len(outputs) != len(dataset.img_infos):
        raise RuntimeError(
            f"Mismatch: outputs={len(outputs)}, dataset={len(dataset.img_infos)}"
        )

    for pred, img_info in zip(outputs, dataset.img_infos):
        fname = os.path.basename(img_info["filename"])

        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = np.asarray(pred)

        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[:, :, 0]

        if pred.ndim != 2:
            raise RuntimeError(
                f"Prediction must be 2D. Got shape={pred.shape} for {fname}"
            )

        # LiTS17 tumor-only:
        # pred > 0 means tumor class after binary training.
        mask = (pred > 0).astype(np.uint8) * 255

        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(mask_dir, out_name)

        Image.fromarray(mask).save(out_path)

    return mask_dir


def evaluate_3d(pred_dir, gt_dir, voxel_volume_mm3, require_contiguous=True):
    """
    Evaluate full 3D reconstructed tumor volumes.
    """
    pred_cases = group_slices(pred_dir)
    gt_cases = group_slices(gt_dir)

    pred_ids = set(pred_cases.keys())
    gt_ids = set(gt_cases.keys())

    missing_pred = sorted(gt_ids - pred_ids)
    missing_gt = sorted(pred_ids - gt_ids)

    if missing_pred:
        raise RuntimeError(
            f"Missing predictions for {len(missing_pred)} cases: {missing_pred[:10]}"
        )

    if missing_gt:
        raise RuntimeError(
            f"Missing GT for {len(missing_gt)} cases: {missing_gt[:10]}"
        )

    rows = []

    for case_id in sorted(gt_ids):
        pred_raw = stack_volume(
            case_id,
            pred_cases[case_id],
            require_contiguous=require_contiguous,
        )

        gt = stack_volume(
            case_id,
            gt_cases[case_id],
            require_contiguous=require_contiguous,
        )

        if pred_raw.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch for {case_id}: pred={pred_raw.shape}, gt={gt.shape}"
            )

        raw_dice = dice_score(pred_raw, gt)
        raw_vol = compute_volume_metrics(pred_raw, gt, voxel_volume_mm3)

        row = {
            "case_id": case_id,
            "num_slices": int(gt.shape[0]),
            "height": int(gt.shape[1]),
            "width": int(gt.shape[2]),
            "raw_3d_dice": raw_dice,
            "gt_voxels": raw_vol["gt_voxels"],
            "raw_pred_voxels": raw_vol["pred_voxels"],
            "gt_volume_mm3": raw_vol["gt_volume_mm3"],
            "raw_pred_volume_mm3": raw_vol["pred_volume_mm3"],
            "raw_abs_volume_error_mm3": raw_vol["abs_volume_error_mm3"],
            "raw_relative_volume_error": raw_vol["relative_volume_error"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df):
    finite_raw_rel = df["raw_relative_volume_error"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    summary = {
        "num_cases": int(len(df)),
        "mean_raw_3d_dice": float(df["raw_3d_dice"].mean()),
        "std_raw_3d_dice": float(df["raw_3d_dice"].std()),
        "mean_raw_abs_volume_error_mm3": float(
            df["raw_abs_volume_error_mm3"].mean()
        ),
        "mean_raw_relative_volume_error": float(finite_raw_rel.mean()),
    }

    return summary


def print_summary(summary):
    print("\n========== LiTS17 CT 3D TEST SUMMARY ==========")
    print(f"Cases: {summary['num_cases']}")
    print(f"Mean raw 3D Dice:       {summary['mean_raw_3d_dice']:.4f}")
    print(f"Std raw 3D Dice:        {summary['std_raw_3d_dice']:.4f}")
    print(f"Mean raw abs vol error: {summary['mean_raw_abs_volume_error_mm3']:.2f} mm3")
    print(f"Mean raw rel vol error: {summary['mean_raw_relative_volume_error']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run MMSeg checkpoint prediction on LiTS17 CT liver tumor test set, "
            "reconstruct 3D tumor volumes, compute raw 3D Dice and tumor volume metrics."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="MMSeg config file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint .pth to evaluate.",
    )

    parser.add_argument(
        "--gt-dir",
        required=True,
        help=(
            "Ground-truth binary tumor mask folder for LiTS17 test split. "
            "Expected preprocessing: mask = (label > 1).astype(uint8)."
        ),
    )

    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory to save/read MMSeg predictions.",
    )

    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for per-case 3D results.",
    )

    parser.add_argument(
        "--voxel-volume-mm3",
        type=float,
        default=1.0,
        help=(
            "Voxel volume in mm^3. Use 1.0 if LiTS17 slices were resampled "
            "to isotropic spacing during preprocessing."
        ),
    )

    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip MMSeg prediction and only evaluate existing pred-dir/preds_mask.",
    )

    parser.add_argument(
        "--no-clean-pred-dir",
        action="store_true",
        help="Do not remove pred-dir before prediction.",
    )

    parser.add_argument(
        "--allow-noncontiguous",
        action="store_true",
        help="Allow non-contiguous slices. Not recommended for 3D Dice/volume.",
    )

    args = parser.parse_args()

    mask_pred_dir = os.path.join(args.pred_dir, "preds_mask")

    if not args.skip_predict:
        mask_pred_dir = run_mmseg_prediction(
            config=args.config,
            checkpoint=args.checkpoint,
            pred_dir=args.pred_dir,
            clean_pred_dir=not args.no_clean_pred_dir,
        )

    df = evaluate_3d(
        pred_dir=mask_pred_dir,
        gt_dir=args.gt_dir,
        voxel_volume_mm3=args.voxel_volume_mm3,
        require_contiguous=not args.allow_noncontiguous,
    )

    # if os.path.dirname(args.out_csv):
    #     os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # df.to_csv(args.out_csv, index=False)

    summary = summarize(df)
    print_summary(summary)


if __name__ == "__main__":
    main()