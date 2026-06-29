#!/bin/bash
# ####### train SSL models ##########
path1="./models/weights/convnext_tiny_1k_224_ema.pth" #for ConvNext-Tiny
path2="./models/weights/mit_b0.pth" #for MiT-B0
root=DAGM_SSL_25 #change to the root directory of the dataset you want to train on

python ./baseline_trainer.py \
    --model "MiT-B0" \
    --init_weight $path2 \
    --training_type "supervised" \
    --dataset "DAGM" \
    --num_classes 7 \
    --base_lr 0.001 \
    --batch_size 12 \
    --num_epochs 350 \
    --unlabeled_ratio 0.0001 \
    --train_img_path "./$root/train/train_images/" \
    --train_mask_path "./$root/train/train_annot/" \
    --test_img_path "./$root/test/test_images/" \
    --test_mask_path "./$root/test/test_annot/" \