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

# TENT-DivGate-Continual on ACDC with 19 classes merged into 10 OVSS super-classes.
# CPU thread limits — avoid OpenCV/MKL spawning hundreds of threads
# when many experiments share 256 cores (res=11 spawn errors).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

# Hyperparameters match save/ACDCDataset/tent_divgate_continual_caut_0.005_brake_0.02 baseline.
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

METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_THRESHOLD=1.8
H_WARNING=1.5
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.005
BRAKE_RST=0.02

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_caut_0.005_brake_0.02/"

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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --save_dir $SAVE_DIR \
                        --class_extensions
