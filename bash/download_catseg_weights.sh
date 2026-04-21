#!/bin/bash
# Download CAT-Seg pretrained weights

SAVE_DIR=".weights/catseg"
mkdir -p $SAVE_DIR

echo "Downloading CAT-Seg (B) ViT-B/16 weights..."
wget -c https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth \
     -O ${SAVE_DIR}/model_base.pth

echo "Downloading CAT-Seg (L) ViT-L/14 weights..."
wget -c https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth \
     -O ${SAVE_DIR}/model_large.pth

echo "Done. Weights saved to ${SAVE_DIR}/"
ls -lh ${SAVE_DIR}/
