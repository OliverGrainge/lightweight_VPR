#!/bin/bash


git add -u 
git commit -m "Adding Results"
git push


dataset=pitts30k
bb=mobilenetv2conv4

#### efficientnet ####
agg=gem
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_4096/best_model.pth
fc_output_dim=4096
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push



dataset=nordland
bb=mobilenetv2conv4

#### efficientnet ####
agg=gem
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=4096
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_4096/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push




dataset=st_lucia
bb=mobilenetv2conv4

#### efficientnet ####
agg=gem
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=4096
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_4096/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push



























dataset=pitts30k
bb=mobilenetv2conv4

#### efficientnet ####
agg=mac
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=4096
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_4096/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push



dataset=nordland
bb=mobilenetv2conv4

#### efficientnet ####
agg=mac
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_4096/best_model.pth
fc_output_dim=4096
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push




dataset=st_lucia
bb=mobilenetv2conv4

#### efficientnet ####
agg=mac
fc_output_dim=512
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_512/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=512 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=1024
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=2048
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_2048/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=2048 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False
fc_output_dim=4096
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_4096/best_model.pth
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=4096 --mixed_precision_training=True --infer_batch_size=1 --num_workers=16 --QAT=False

git add -u 
git commit -m "Adding Results"
git push

























