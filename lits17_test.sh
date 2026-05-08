python tools_med/lits17_postprocess.py \
  --config local_configs/segformer/B1/segformer.b1.512x512.lits17.py \
  --checkpoint work_dirs/liver_b1/trains/iter_80000.pth \
  --gt-dir data/medical_seg/liver/annotations/test \
  --pred-dir work_dirs/liver_b1/preds \
  --out-csv work_dirs/liver_b1/preds/results_3d.csv