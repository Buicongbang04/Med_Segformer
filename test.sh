python tools_med/postprocessing_brats20.py \
  --config local_configs/segformer/B1/segformer.b1.512x512.brats20.py \
  --checkpoint work_dirs/brats20_b1/trains/iter_40000.pth \
  --gt-dir data/medical_seg/brain/annotations/test \
  --pred-dir work_dirs/brats20_b1/preds \
  --out-csv work_dirs/brats20_b1/preds/3d_results.csv \
  --voxel-volume-mm3 1.0