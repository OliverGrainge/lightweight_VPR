#!/bin/bash


##### AGGREGATION IS GEM ####################################

python train.py --backbone=mobilenetv2conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_512 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=512
python train.py --backbone=mobilenetv2conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_2048 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=2048
python train.py --backbone=mobilenetv2conv4 --aggregation=gem --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_gem_4096 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=4096




python train.py --backbone=mobilenetv2conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_mac_512 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=512
python train.py --backbone=mobilenetv2conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_mac_2048 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=2048
python train.py --backbone=mobilenetv2conv4 --aggregation=mac --dataset_name=mapillary_sls --num_workers=16 --save_dir=mobilenetv2conv4_mac_4096 --QAT=False --train_batch_size=2 --infer_batch_size=1 --mixed_precision_training=True --accrue_gradient=1 --fc_output_dim=4096



