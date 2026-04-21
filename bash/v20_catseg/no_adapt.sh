# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="384 384"
ALL_CORRUPTIONS="original gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
OVSS_TYPE="catseg"
OVSS_BACKBONE="ViT-L/"14
CATSEG_CKPT=".weights/catseg/model_large.pth"

# Hyperparameters
BATCH_SIZE=2
LR=0.001
STEPS=10
TRIALS=3

# Output
SAVE_DIR=".save/${DATASET}_catseg/No_Adaptation/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
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
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --plot_loss \
                        --class_extensions
