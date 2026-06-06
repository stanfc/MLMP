#!/bin/bash
GPU_ID=${GPU_ID:-3}
N_IMAGES=5
SEED=42
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

run() {
    DATASET=$1
    DATA_DIR=$2
    INIT_RESIZE=$3
    EXTRA=${4:-""}

    echo ""
    echo "=============================="
    echo "Dataset: $DATASET"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=$GPU_ID python qualitative.py \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --save_dir .qualitative/${DATASET}/ \
        --n_images $N_IMAGES \
        --seed $SEED \
        --ovss_type $OVSS_TYPE \
        --ovss_backbone $OVSS_BACKBONE \
        --prompt_dir $PROMPT_DIR \
        --vision_outputs $OUT_VISION \
        --alpha_cls 1.0 \
        --lr 0.001 \
        --steps 10 \
        --init_resize $INIT_RESIZE \
        --patch_size 224 224 \
        --patch_stride 112 \
        --class_extensions \
        $EXTRA
}

# Parse --legend_only flag
EXTRA=""
for arg in "$@"; do
    if [ "$arg" = "--legend_only" ]; then
        EXTRA="--legend_only"
    fi
done

run COCOStuffDataset       .data/coco_stuff164k/  "224 224"  "$EXTRA"
run COCOObjectDataset      .data/coco_object/     "224 224"  "$EXTRA"
run CityscapesDataset      .data/cityscapes/      "1120 560" "$EXTRA"
run PascalVOC20Dataset     .data/VOC2012/         "224 224"  "$EXTRA"
run PascalVOC21Dataset     .data/VOC2012/         "224 224"  "$EXTRA"
run PascalContext59Dataset .data/VOC2010/         "224 224"  "$EXTRA"
run PascalContext60Dataset .data/VOC2010/         "224 224"  "$EXTRA"
