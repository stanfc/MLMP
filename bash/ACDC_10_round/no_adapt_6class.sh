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

# No-Adaptation baseline on ACDC with 19 classes merged into 6 OVSS super-classes.
# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

# Class names directly used as CLIP prompts (see utils/class_extensions/acdc_6class.txt).
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

GPU_ID=1

DATASET=ACDCMerged6Dataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
METHOD="tent_continual"

CONTINUAL_ROUNDS=10
BATCH_SIZE=1

SAVE_DIR="save/${DATASET}/No_Adaptation/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        --save_dir $SAVE_DIR \
                        --class_extensions
