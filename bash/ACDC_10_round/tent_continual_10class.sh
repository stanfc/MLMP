# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

#!/bin/bash
# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

# TENT-Continual on ACDC with 19 classes merged into 10 OVSS super-classes.
# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2


# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=ACDCMerged10Dataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        --save_dir $SAVE_DIR \
                        --class_extensions
