FOLD_NUM=4
ITER=20000

while true
do
  echo "Use checkpoint iter_${ITER}.pth for fold ${FOLD_NUM}"
  python tools_med/brats20_postprocess.py \
    --config local_configs/segformer/B1/segformer.b1.512x512.brats20.py \
    --checkpoint work_dirs/cross_validation/brats20/fold_${FOLD_NUM}/trains/iter_${ITER}.pth \
    --gt-dir data/cross_validation/brats20/fold_${FOLD_NUM}/annotations/test \
    --pred-dir work_dirs/cross_validation/brats20/fold_${FOLD_NUM}/preds \
    --out-csv work_dirs/cross_validation/brats20/fold_${FOLD_NUM}/preds/3d_results.csv \
    --voxel-volume-mm3 1.0

  echo "Postprocessing completed for fold ${FOLD_NUM} with checkpoint iter_${ITER}.pth"
  echo "===================================================================="

  # read -p "Do you want to evaluate another checkpoint? (y/n) " answer

  # if [[ "$answer" != "y" ]]; then
  #   break
  # fi

  ITER=$((ITER - 4000))

  if [ $ITER -lt 4000 ]; then
    echo "No more checkpoints to evaluate."
    break
  fi

  echo "Continue to evaluate checkpoint iter_${ITER}.pth for fold ${FOLD_NUM}?"
done