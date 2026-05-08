python tools_med/inference_brats20.py \
  --nifti /home/bangbc/Documents/CapstoneProject/ThesisProject26_v2/data/BraTS20/MICCAI_BraTS2020_TrainingData/BraTS20_Training_017/BraTS20_Training_017_flair.nii \
  --config local_configs/segformer/B1/segformer.b1.512x512.brats20.py \
  --checkpoint work_dirs/brats20_b1/trains/iter_40000.pth \
  --out-root work_dirs/brats20_b1/infer \
  --device cuda:0 \
  --clean