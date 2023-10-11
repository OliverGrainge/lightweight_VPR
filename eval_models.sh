#!/bin/bash

dataset=pitts30k

#### efficientnet ####
bb=efficientnet_b0
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push















dataset=nordland

#### efficientnet ####
bb=efficientnet_b0
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push









dataset=st_lucia

#### efficientnet ####
bb=efficientnet_b0
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=netvlad
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_netvlad_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push






































dataset=st_lucia

#### efficientnet ####
bb=efficientnet_b0
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push




dataset=pitts30k

#### efficientnet ####
bb=efficientnet_b0
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push









dataset=nordland

#### efficientnet ####
bb=efficientnet_b0
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=spoc
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push






































dataset=st_lucia

#### efficientnet ####
bb=efficientnet_b0
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push





dataset=pitts30k

#### efficientnet ####
bb=efficientnet_b0
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push





dataset=nordland

#### efficientnet ####
bb=efficientnet_b0
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_gem_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=gem
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_spoc_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push






















dataset=nordland

#### efficientnet ####
bb=efficientnet_b0
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push




dataset=st_lucia

#### efficientnet ####
bb=efficientnet_b0
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push




dataset=pitts30k

#### efficientnet ####
bb=efficientnet_b0
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet18 ####
bb=resnet18conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### resnet50 ####
bb=resnet50conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### mobilenet ####
bb=mobilenetv2conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024


git add -u 
git commit -m "Adding Results"
git push

#### squeezenet ####
bb=squeezenet
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### VGG16 ####
bb=vgg16
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/vgg16_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push

#### RESNET101 ####
bb=resnet101conv4
agg=mac
resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet101_mac_1024/best_model.pth

python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp32_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=fp16_comp --resume=$resume --fc_output_dim=1024
python eval.py --backbone=$bb --aggregation=$agg --dataset_name=$dataset --precision=int8_comp --resume=$resume --fc_output_dim=1024

git add -u 
git commit -m "Adding Results"
git push































