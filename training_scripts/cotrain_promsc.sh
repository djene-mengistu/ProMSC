#!/bin/bash
# ####### train SSL models ##########
path1="./models/weights/convnext_tiny_1k_224_ema.pth"
path2="./models/weights/mit_b0.pth"
root=NEU_VOC

python ./cotrain_promsc.py \
    --model1 "MiT-B0" \
    --model2 "ConvNeXt-T" \
    --init_weight_1 $path2 \
    --init_weight_2 $path1 \
    --training_type "SSL" \
    --dataset "NEU" \
    --num_classes 4 \
    --base_lr 0.001 \
    --batch_size 4 \
    --num_epochs 600 \
    --unlabeled_ratio 0.9 \
    --train_img_path "./$root/train/train_images/" \
    --train_mask_path "./$root/train/train_annot/" \
    --test_img_path "./$root/test/test_images/" \
    --test_mask_path "./$root/test/test_annot/" \