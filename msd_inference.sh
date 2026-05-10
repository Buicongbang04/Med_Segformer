python tools_med/msd07_inference.py \
  --nifti data/Pancreas/imagesTr/pancreas_050.nii.gz \
  --config local_configs/segformer/B1/segformer.b1.512x512.msd07.py \
  --checkpoint work_dirs/msd07_b1/trains/iter_60000.pth \
  --out-root work_dirs/msd07_b1/infer \
  --device cuda:0 \
  --clean