#!/bin/bash


# training shufflenet: complete
#python train.py --backbone=shufflenetv2 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=shufflenet_gem_1024 --QAT=False

# training efficientnet: complete
#python train.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_gem_1024 --QAT=False

# training shufflenet 
#python train.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_gem_1024 --QAT=False
# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_1024 --QAT=False

# training resnet18
#python train.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_gem_1024 --QAT=False

# training resnet50
#python train.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_1024 --QAT=False

python train.py --backbone=vgg16 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_gem_1024 --QAT=False

python train.py --backbone=resnet101conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_gem_1024 --QAT=False



######################################### Train with NetVLAD #################################################################################################


# training efficientnet: complete
python train.py --backbone=efficientnet_b0 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_netvlad_1024 --QAT=False

# training shufflenet 
python train.py --backbone=squeezenet --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_netvlad_1024 --QAT=False

# training mobilenet
python train.py --backbone=mobilenetv2conv4 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_netvlad_1024 --QAT=False

# training resnet18
python train.py --backbone=resnet18conv4 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_netvlad_1024 --QAT=False

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_netvlad_1024 --QAT=False

python train.py --backbone=vgg16 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_netvlad_1024 --QAT=False

python train.py --backbone=resnet101conv4 --aggregation=netvlad --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_netvlad_1024 --QAT=False




######################################### Train with CRN #################################################################################################


# training efficientnet: complete
python train.py --backbone=efficientnet_b0 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_crn_1024 --QAT=False

# training shufflenet 
python train.py --backbone=squeezenet --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_crn_1024 --QAT=False

# training mobilenet
python train.py --backbone=mobilenetv2conv4 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_crn_1024 --QAT=False

# training resnet18
python train.py --backbone=resnet18conv4 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_crn_1024 --QAT=False

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_crn_1024 --QAT=False

python train.py --backbone=vgg16 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_crn_1024 --QAT=False

python train.py --backbone=resnet101conv4 --aggregation=crn --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_crn_1024 --QAT=False





##############################################################################################################################################
################################################ QAT #########################################################################################
##############################################################################################################################################

# training shufflenet: complete
#python train.py --backbone=shufflenetv2 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=shufflenet_gem_1024 --QAT=False

# training efficientnet: complete
#python train.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_gem_1024 --QAT=False

# training shufflenet 
#python train.py --backbone=squeezenet --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_gem_1024 --QAT=False
# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_1024 --QAT=False

# training resnet18
#python train.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_gem_1024 --QAT=False

# training resnet50
#python train.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_1024 --QAT=False

#python train.py --backbone=vgg16 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_gem_1024 --QAT=False

#python train.py --backbone=resnet101conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_gem_1024 --QAT=False