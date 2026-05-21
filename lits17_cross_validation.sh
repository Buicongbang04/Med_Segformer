# !/bin/bash
# Path to split folder
SPLIT_FOLD="data/splits/lits17"

for i in 1
do
    FILE_NAME="liver_fold_$i.json"
    echo "Processing data for split file: $FILE_NAME"

    # Run preprocessing script
    python tools_med/lits17_preprocess.py \
    --split-file $SPLIT_FOLD/$FILE_NAME \
    --out-root data/cross_validation/lits17/fold_$i 

    echo "Preprocessing completed !!!"

    echo "==============================================================================================================================="
    echo "Starting training for fold $i"
    # Run training script
    python tools/train.py \
    local_configs/segformer/B1/segformer.b1.512x512.lits17.py \
    --work-dir work_dirs/cross_validation/lits17/fold_$i/trains
done