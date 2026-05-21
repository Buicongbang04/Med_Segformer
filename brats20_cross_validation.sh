# !/bin/bash
# Path to split folder
SPLIT_FOLD="data/splits/brats20"

for i in 0 1 2 3 4
do
    FILE_NAME="brain_fold_$i.json"
    echo "Processing data for split file: $FILE_NAME"

    # Run preprocessing script
    python tools_med/brats20_preprocess.py \
        --split-file $SPLIT_FOLD/$FILE_NAME \
        --out-root data/cross_validation/brats20/fold_$i 

    echo "Preprocessing completed !!!"

    echo "==============================================================================================================================="
    echo "Starting training for fold $i"
    # Run training script
    python tools/train.py \
        local_configs/segformer/B1/segformer.b1.512x512.brats20.py \
        --work-dir work_dirs/cross_validation/brats20/fold_$i/trains \
        --options \
            data.train.data_root=data/cross_validation/brats20/fold_${i} \
            data.val.data_root=data/cross_validation/brats20/fold_${i} \
            data.test.data_root=data/cross_validation/brats20/fold_${i}
done