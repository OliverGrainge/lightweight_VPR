#!/bin/bash

dataset=pitts30k

#### efficientnet ####
bb=mobilenetv2conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push























