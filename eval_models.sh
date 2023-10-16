#!/bin/bash

dataset=pitts30k


#### efficientnet ####
bb=resnet101conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_netvlad_1024/best_model.pth
#python eval.py --backbone=resnet101conv4 --aggregation=gem --dataset_name=$dataset --precision=fp32_comp --num_workers=16 --save_dir=resnet101_gem_1024 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push



























