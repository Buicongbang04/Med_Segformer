python tools_med/lits17_inference.py \
  --nifti data/LITS17/volume-100.nii \
  --config local_configs/segformer/B1/segformer.b1.512x512.lits17.py \
  --checkpoint work_dirs/liver_b1/trains/iter_80000.pth \
  --out-root work_dirs/liver_b1/infer \
  --device cuda:0 \
  --clean