#!/bin/bash

#### efficientnet ####

#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp32_comp


#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=fp16_comp


#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/efficientnetb0_gem_1024/best_model.pth --precision=int8_comp


#### resnet18 ####
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp32_comp


#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=fp16_comp


#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet18_gem_1024/best_model.pth --precision=int8_comp




#### resnet50 ####
#python eval.py --backbone=resnet50conv4  --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp32_comp


#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=fp16_comp


#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/resnet50_gem_1024/best_model.pth --precision=int8_comp



#### mobilenet ####
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp32_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp32_comp


#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp16_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=fp16_comp


#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2_gem_1024/best_model.pth --precision=int8_comp




#### squeezenet ####
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp16_comp
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp16_comp
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp32_comp
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp32_comp

python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp32_comp
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp32_comp

python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp32_comp


python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp16_comp
python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp16_comp


python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=fp16_comp


#python eval.py --backbone=squeezenet--aggregation=gem --fc_output_dim=1024 --dataset_name=eynsham --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=nordland --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=pitts30k --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=int8_comp
#python eval.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=st_lucia --resume=/home/oliver/Documents/github/lightweight_VPR/weights/fp32/squeezenet_gem_1024/best_model.pth --precision=int8_comp