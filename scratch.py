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

args = parser.parse_arguments()

args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.dataset_name = "pitts30k"
args.infer_batch_size=3
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.fc_output_dim = 1024
args.device = 'cuda:0'
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{args.aggregation}_{str(args.fc_output_dim)}/best_model.pth"
model = network.GeoLocalizationNet(args).eval().to(args.device)
model = util.resume_model(args, model)
model.eval().cuda()
dummy_input = torch.randn(1, 3, 480, 640, device='cuda')


onnx_file_path = 'model.onnx'

torch.onnx.export(model, dummy_input, onnx_file_path, input_names=['input'],
                  output_names=['output'], export_params=True)
# Load your ONNX model


onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)
print(onnx)
# Set up the TensorRT logger and create a builder and network


TRT_LOGGER = trt.Logger()

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)


with open(onnx_file_path, 'rb') as model:
    print('Beginning ONNX file parsing')
    parser.parse(model.read())
print('Completed parsing of ONNX file')

layer_count = 0
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.type == trt.LayerType.CONVOLUTION:
        layer.set_precision(trt.float16)
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float16)
        layer_count += 1
    
    print(f"Layer {i}: {layer.name}, type: {layer.type}, count_layer {layer_count}")


print(network.num_layers)
config = builder.create_builder_config()
trt_model_engine = builder.build_serialized_network(network, config)
#trt_model_context = trt_model_engine.create_execution_context()
