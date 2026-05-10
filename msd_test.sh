python tools_med/msd07_postprocess.py \
  --config local_configs/segformer/B1/segformer.b1.512x512.msd07.py \
  --checkpoint work_dirs/msd07_b1/trains/iter_60000.pth \
  --gt-dir data/medical_seg/msd/annotations/test \
  --pred-dir work_dirs/msd07_b1/preds \
  --out-csv work_dirs/msd07_b1/preds/results_3d.csv