# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1152 768"
ALL_CORRUPTIONS="original gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="watt"
PROMPT_DIR="prompts.yaml"
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/14"
CATSEG_CKPT=".weights/catseg/model_large.pth"

# Hyperparameters
BATCH_SIZE=2
LR=0.001
WATT_L=2
WATT_M=5
TRIALS=3

# Output
SAVE_DIR=".save/${DATASET}_catseg/${METHOD}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --method $METHOD \
                        --prompt_dir $PROMPT_DIR \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone "$OVSS_BACKBONE" \
                        $( [ -n "$CATSEG_CKPT" ] && echo "--catseg_checkpoint $CATSEG_CKPT" ) \
                        \
                        --save_dir $SAVE_DIR \
                        --data_dir $DATA_DIR \
                        --dataset $DATASET \
                        --workers $WORKERS \
                        --init_resize $INIT_RESIZE \
                        --patch_size 384 384 \
                        --patch_stride 192 \
                        --corruptions_list $ALL_CORRUPTIONS \
                        \
                        --lr $LR \
                        --watt_l $WATT_L \
                        --watt_m $WATT_M \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --plot_loss \
                        --class_extensions
