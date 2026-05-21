FOLD_NUM=1
ITER=4000

while true
do
  echo "Use checkpoint iter_${ITER}.pth for fold ${FOLD_NUM}"

  python tools_med/lits17_postprocess.py \
    --config local_configs/segformer/B1/segformer.b1.512x512.lits17.py \
    --checkpoint work_dirs/cross_validation/lits17/fold_${FOLD_NUM}/trains/iter_${ITER}.pth \
    --gt-dir data/cross_validation/lits17/fold_${FOLD_NUM}/annotations/test \
    --pred-dir work_dirs/cross_validation/lits17/fold_${FOLD_NUM}/preds \
    --out-csv work_dirs/cross_validation/lits17/fold_${FOLD_NUM}/preds/results_3d.csv

  echo "Postprocessing completed for fold ${FOLD_NUM} with checkpoint iter_${ITER}.pth"
  echo "===================================================================="

  read -p "Do you want to evaluate another checkpoint? (y/n) " answer

  if [[ "$answer" != "y" ]]; then
    break
  fi

  ITER=$((ITER + 4000))

  echo "Continue to evaluate checkpoint iter_${ITER}.pth for fold ${FOLD_NUM}?"
done