python tools/test.py \
  local_configs/segformer/B1/segformer.b1.512x512.brats20.py \
  work_dirs/brats20_b1/best_checkpoint.pth \
  --eval mDice \
  --show-dir work_dirs/brats20_b1/preds