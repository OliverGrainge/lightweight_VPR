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

#python train.py --backbone=vgg16 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_gem_1024 --QAT=False

#python train.py --backbone=resnet101conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_gem_1024 --QAT=False

######################################### Train with rrm #################################################################################################

#train vgg16
#python train.py --backbone=vgg16 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resume=/home/oliver/Documents/github/lightweight_VPR/logs/vgg16_rrm_1024/2023-09-20_06-27-36/best_model.pth

#train resnet101
#python train.py --backbone=resnet101conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024

# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_rrm_1024 --QAT=False --train_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --infer_batch_size=3 --fc_output_dim=1024

# training resnet50
#python train.py --backbone=resnet50conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=Frue --accrue_gradient=1 --fc_output_dim=1024

# training efficientnet:
#python train.py --backbone=efficientnet_b0 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --accrue_gradient=1 --mixed_precision_training=False --fc_output_dim=1024

# training resnet18
#python train.py --backbone=resnet18conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --accrue_gradient=1 --mixed_precision_training=False --fc_output_dim=1024 --resume=/home/oliver/Documents/github/lightweight_VPR/logs/resnet18_rrm_1024/2023-09-21_06-42-28/best_model.pth

# training shufflenet 
#python train.py --backbone=squeezenet --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_rrm_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024

######################################### Train with CRN #################################################################################################


#train vgg16
#python train.py --backbone=vgg16 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=vgg16_rmac_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024

#train resnet101
#python train.py --backbone=resnet101conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet101_rmac_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024

# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_rmac_1024 --QAT=False --train_batch_size=3 --mixed_precision_training=False --accrue_gradient=1 --infer_batch_size=3 --fc_output_dim=1024

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rmac_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --mixed_precision_training=Frue --accrue_gradient=1 --fc_output_dim=1024

# training efficientnet:
#python train.py --backbone=efficientnet_b0 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_rmac_1024 --QAT=False --train_batch_size=3 --infer_batch_size=3 --accrue_gradient=1 --mixed_precision_training=False --fc_output_dim=1024

# training resnet18
#python train.py --backbone=resnet18conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_rmac_1024 --QAT=False --train_batch_size=5 --infer_batch_size=4 --accrue_gradient=1 --mixed_precision_training=False --fc_output_dim=1024

# training shufflenet 
#python train.py --backbone=squeezenet --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=squeezenet_rmac_1024 --QAT=False --train_batch_size=5 --infer_batch_size=4 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024



##################################### Train Resolution Models #########################################################################################

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_res128 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=128 128
python train.py --backbone=resnet50conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_res256 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=256 256
python train.py --backbone=resnet50conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_res512 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=512 512
python train.py --backbone=resnet50conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_res1024 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=1024 1024

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rmac_res128 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=128 128 
python train.py --backbone=resnet50conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rmac_res256 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=256 256
python train.py --backbone=resnet50conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rmac_res512 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=512 512
python train.py --backbone=resnet50conv4 --aggregation=rmac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rmac_res1024 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=1024 1024

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rrm_res128 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=128 128
python train.py --backbone=resnet50conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rrm_res256 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=256 256
python train.py --backbone=resnet50conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rrm_res512 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=512 512
python train.py --backbone=resnet50conv4 --aggregation=rrm --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_rrm_res1024 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=1024 1024

# training resnet50
python train.py --backbone=resnet50conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_mac_res128 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=128 128
python train.py --backbone=resnet50conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_mac_res256 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=256 256
python train.py --backbone=resnet50conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_mac_res512 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=512 512
python train.py --backbone=resnet50conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_mac_res1024 --QAT=False --train_batch_size=2 --infer_batch_size=2 --mixed_precision_training=False --accrue_gradient=1 --fc_output_dim=1024 --resize=1024 1024















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