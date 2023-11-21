import torch 
import tensorrt as trt
import onnx

from QuantizationSim import Quantizer 
from torchvision.models import resnet18
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn 
from model import network
import util
import datasets_ws
import parser
import numpy as np
import test
import torch.nn.functional as F
import copy
import time 
import os
import pandas as pd

args = parser.parse_arguments()

args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.dataset_name = "nordland"
args.infer_batch_size=5
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.fc_output_dim = 1024
args.device = 'cuda:0'
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{args.aggregation}_{str(args.fc_output_dim)}/best_model.pth"
model = network.GeoLocalizationNet(args).eval().to(args.device)
model = util.resume_model(args, model)
model.eval().cuda()
dummy_input = torch.randn(1, 3, 480, 640, device='cuda')
NUM_SAMPLES = 300
BATCH_SIZE=5

average_bitwidths = [14, 13, 12, 11, 10, 9, 8]
average_bitwidths.reverse()
print(average_bitwidths)

def convert_config(config, layer_types):
    new_config = []
    i = 0
    idx = 0
    idx = 0
    while len(new_config) < len(layer_types):
        if layer_types[idx].startswith('Conv2d') and layer_types[idx+1].startswith('BatchNorm2d'):
            new_config.append(config[i])
            new_config.append(config[i])
            idx += 2
            i += 1
        elif layer_types[idx] == "Linear":
            new_config.append(config[i])
            idx += 1
            i += 1
        elif layer_types[idx].startswith('Conv2d') and layer_types[idx + 1].startswith('Conv2d'):
            new_config.append(config[i])
            idx += 1
            i += 1
    return new_config

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
# Generate random indices

# Create a subset
test_dl = DataLoader(test_ds, BATCH_SIZE)

quantizer = Quantizer(model,
        layer_precision="int8",
        activation_precision="fp32",
        activation_granularity="tensor",
        layer_granularity="channel",
        calibration_type="minmax", 
        cal_samples=20,
        calibration_loader=test_dl,
        activation_layers = (nn.Conv2d, nn.BatchNorm2d, nn.Linear), 
        device=args.device, 
        number_of_activations=52,
        number_of_layers=103)

quantizer.fit()


layer_types = quantizer.all_layer_types


all_recalls = []
for bitwidth in average_bitwidths:
    config = np.load(f"/home/oliver/Documents/github/lightweight_VPR/evoquant_data/mobilenetv2conv4_mac_1024_{str(bitwidth)}/best_configuration.npy")
    full_layer_config = convert_config(config, layer_types)
    int4_mask = config == "int4"
    int8_mask = config == "int8"
    activation_config = [str(i) for i in full_layer_config]
    for i in range(len(activation_config)):
        if activation_config[i] == "int4":
            activation_config[i] == "int8"
        if activation_config[i] == "int8":
            activation_config[[i] == "int32"]

    quantizer = Quantizer(model,
        layer_configuration=full_layer_config,
        activation_configuration=list(activation_config),
        activation_granularity="tensor",
        layer_granularity="channel",
        calibration_type="minmax", 
        cal_samples=15,
        calibration_loader=test_dl,
        activation_layers = (nn.Conv2d, nn.BatchNorm2d, nn.Linear), 
        device=args.device, 
        number_of_activations=52,
        number_of_layers=103)
    
    qmodel = quantizer.fit()
    print(qmodel)
    del quantizer

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, qmodel, args.test_method, None
    )
    del qmodel
    del test_ds
    print(recalls_str)
    all_recalls.append(recalls)
    print(recalls)
    print(type(recalls))


all_recalls = np.array(all_recalls)
results_df = pd.DataFrame.from_dict({"recall@1":all_recalls[:, 0], 
                                     "recall@5":all_recalls[:, 1], 
                                     "recall@10":all_recalls[:, 2],
                                     "average_bitwidth": average_bitwidths})



results_df.to_csv("evo_quant_line_plot_nordland.csv")