
"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.
The first is https://github.com/naver/deep-image-retrieval from Naver labs,
provides ResNet-50 and ResNet-101 trained with AP on Google Landmarks 18 clean.
$ python eval.py --off_the_shelf=naver --l2=none --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

The second is https://github.com/filipradenovic/cnnimageretrieval-pytorch from
Radenovic, provides ResNet-50 and ResNet-101 trained with a triplet loss
on Google Landmarks 18 and sfm120k.
$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048
$ python eval.py --off_the_shelf=radenovic_sfm --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


import sys
import torch
import parser
import logging
import sklearn
from torch import nn
import torch_tensorrt 
from torch.cuda.amp import autocast
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd
from util import count_parameters, quantize_model, measure_memory

import test
import pandas as pd
import util
import commons
import datasets_ws
from model import network
from util import benchmark_latency

OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
df = pd.read_csv("results.csv", index_col="id")

if args.resume is not None:
    QAT_FLAG = "PTQ"
    if "int8" in args.resume or "fp16" in args.resume:
        QAT_FLAG = "QAT"
else: 
    QAT_FLAG = "PTQ"

try:
    result_data = df.loc[args.backbone + "_" + args.aggregation + "_" + args.precision + "_" + QAT_FLAG + "_" + str(args.fc_output_dim)].to_dict()
except: 
    result_data = {}

######################################### MODEL #########################################
model = network.GeoLocalizationNet(args).cuda().eval()

if args.aggregation in ["netvlad", "crn"]:
    args.features_dim *= args.netvlad_clusters

if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
    if args.off_the_shelf.startswith("radenovic"):
        pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
        url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
        state_dict = load_url(url, model_dir=join("data", "off_the_shelf_nets"))
    else:
        # This is a hacky workaround to maintain compatibility
        sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
        zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
        if not os.path.exists(zip_file_path):
            gdd.download_file_from_google_drive(file_id=OFF_THE_SHELF_NAVER[args.backbone],
                                                dest_path=zip_file_path, unzip=True)
        if args.backbone == "resnet50conv5":
            state_dict_filename = "Resnet50-AP-GeM.pt"
        elif args.backbone == "resnet101conv5":
            state_dict_filename = "Resnet-101-AP-GeM.pt"
        state_dict = torch.load(join("data", "off_the_shelf_nets", state_dict_filename))
    state_dict = state_dict["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)
elif args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
if args.fc_output_dim is not None:
    args.features_dim = args.fc_output_dim

result_data["id"] = args.backbone + "_" + args.aggregation + "_" + args.precision + "_" + QAT_FLAG + "_" + str(args.fc_output_dim)
result_data["model_path"] = args.resume
result_data["fc_output_dim"] = args.fc_output_dim
result_data["backbone"] = args.backbone 
result_data["aggregation"] = args.aggregation
result_data["precision"] = args.precision

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################## COUNT PARAMETERS #################################


n_params = count_parameters(model)

logging.info(f"number of model parameters: {n_params}")
######################################### QUANTIZE #########################################

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

calibration_dataset = Subset(test_ds, list(range(test_ds.database_num, test_ds.database_num+test_ds.queries_num)))
model = quantize_model(model, precision=args.precision, calibration_dataset=calibration_dataset, args=args)


if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)


######################################## Measure Memory footprint ##########################
model_size = measure_memory(model, input_size=(args.infer_batch_size, 3, 480, 640), args=args)
result_data["model_size"] = model_size
logging.info(f"Model size is {model_size} bytes")


######################################### TEST on TEST SET #################################
recalls, retrieval_time, feature_bytes, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
result_data[args.dataset_name + "_r@1"] = recalls[0]
result_data["descriptor_size"] = feature_bytes
result_data[args.dataset_name + "_r@5"] = recalls[1]
result_data[args.dataset_name + "_r@10"] = recalls[2]
result_data[args.dataset_name + "_r@20"] = recalls[3]
result_data["mean_retrieval_time"] = retrieval_time
logging.info(f"Recalls on {test_ds}: {recalls_str}")


######################################## Test Latency ######################################
torch.cuda.empty_cache()
lat_mean, lat_var = benchmark_latency(model, input_size=(3, 480, 640), batch_size=args.infer_batch_size, precision=args.precision)
result_data["mean_encoding_time"] = lat_mean
result_data["std_encoding_time"] = lat_var
result_data["retrieval_time"] = retrieval_time
print("==================================================================")
print(result_data)
logging.info(f"Mean Inference Latency: {lat_mean}, STD Inference Latency: {lat_var}")

df.loc[args.backbone + "_" + args.aggregation + "_" + args.precision + "_" + QAT_FLAG + "_" + str(args.fc_output_dim)] = result_data
df.to_csv("results.csv")

############################ LOGGING INFORMATION ##################################################
logging.info(f"Model Precision: {args.precision}")
logging.info(f"Model Backbone: {args.backbone}")
logging.info(f"Model aggregation: {args.aggregation}")
logging.info(f"Resumed from: {args.resume}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
