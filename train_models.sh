#!/bin/bash


# training shufflenet 
python train.py --backbone=shufflenetv2 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=shufflenet_gem_1024 --QAT=False

# training efficientnet
#python train.py --backbone=efficientnet_b0 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_gem_1024 --QAT=False

# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_1024 --QAT=False

# training resnet18
#python train.py --backbone=resnet18conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_gem_1024 --QAT=False

# training resnet50
#python train.py --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=1024 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_1024 --QAT=False


################################################# Fine tune for 512 output dimensions ######################################
# training shufflenet 
#python train.py --backbone=shufflenetv2 --resume=FILL_IN --aggregation=gem --fc_output_dim=512 --dataset_name=mapillary_sls --num_workers=16 --save_dir=shufflenet_gem_512 --QAT=False

# training efficientnet
#python train.py --backbone=efficientnet_b0 --resume=FILL_IN --aggregation=gem --fc_output_dim=512 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_gem_512 --QAT=False

# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=512 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_512 --QAT=False

# training resnet18
#python train.py --backbone=resnet18conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=512 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_gem_512 --QAT=False

# training resnet50
#python train.py --backbone=resnet50conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=512 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_512 --QAT=False



################################################# Fine tune for 156 output dimensions ######################################
# training shufflenet 
#python train.py --backbone=shufflenetv2 --resume=FILL_IN --aggregation=gem --fc_output_dim=256 --dataset_name=mapillary_sls --num_workers=16 --save_dir=shufflenet_gem_256 --QAT=False

# training efficientnet
#python train.py --backbone=efficientnet_b0 --resume=FILL_IN --aggregation=gem --fc_output_dim=256 --dataset_name=mapillary_sls --num_workers=16 --save_dir=efficientnetb0_gem_256 --QAT=False

# training mobilenet
#python train.py --backbone=mobilenetv2conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=256 --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_256 --QAT=False

# training resnet18
#python train.py --backbone=resnet18conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=256 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet18_gem_256 --QAT=False

# training resnet50
#python train.py --backbone=resnet50conv4 --resume=FILL_IN --aggregation=gem --fc_output_dim=256 --dataset_name=mapillary_sls --num_workers=16 --save_dir=resnet50_gem_256 --QAT=False


############################################### Fine tune with quantization aware training fp16 ###########################



############################################## Fine tune with quantization aware training int8 #############################