import logging
import os
import sys
import test
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt
import torchvision
from hyperopt import STATUS_OK, fmin, hp, tpe
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm
import io

import datasets_ws
import util
from model import network

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import pickle

import parser


def measure_memory(model):
    script0 = model.block0
    script
    
    byte_obj = torch.jit._save_to_buffer(model)
    # Measure the size
    size = len(byte_obj)
    return size

class PartitionedModel(nn.Module):
    def __init__(self, modules):
        super(PartitionedModel, self).__init__()
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

def split_model(model):
    # Assuming that the backbone has 13 InvertedResidual blocks as in the MobileNetV2 architecture
    backbone_partitions = [model.backbone[:4], model.backbone[4:7], model.backbone[7:], model.aggregation[:]]
    # Assuming that the aggregation has a certain number of layers
    partitioned_models = []
    partitioned_models = [PartitionedModel(part) for part in backbone_partitions]
    return partitioned_models


class HybridModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.block0 = models[0]
        self.block1 = models[1]
        self.block2 = models[2]
        self.block3 = models[3]

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

def quantize_model(model, input_shape: tuple, precision: str, calibration_dataset=None, args=None):
    if precision == "fp32":
        sample_input = torch.randn((1,) + input_shape).float().cuda()
        model.cuda()
        traced_script_module = torch.jit.trace(model, sample_input)

        trt_model_fp32 = torch_tensorrt.compile(
            traced_script_module,
            inputs=[
                torch_tensorrt.Input(
                    (args.infer_batch_size,) + input_shape, dtype=torch.float32
                )
            ],
            enabled_precisions=torch.float32,  # Run with FP32
            workspace_size=1 << 22,
        )
        return trt_model_fp32
    
    elif precision == "fp16":
        sample_input = torch.randn((1,) + input_shape).float().cuda()
        model.cuda()
        traced_script_module = torch.jit.trace(model, sample_input)
        trt_model_fp32 = torch_tensorrt.compile(
            traced_script_module,
            inputs=[
                torch_tensorrt.Input(
                    (args.infer_batch_size,) + input_shape, dtype=torch.float32
                )
            ],
            enabled_precisions=torch.half,  # Run with FP32
            workspace_size=1 << 22,
        )
        return trt_model_fp32
    elif precision == "int8":
        if calibration_dataset is None:
            raise Exception("must provide calibration dataset for int8 quantization")
        
        sample_input = torch.randn((1,) + input_shape).float().cuda()
        model.cuda()
        traced_script_module = torch.jit.trace(model, sample_input)

        testing_dataloader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=args.infer_batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(
            traced_script_module,
            inputs=[torch_tensorrt.Input((args.infer_batch_size,) + input_shape)],
            enabled_precisions={torch.int8},
            calibrator=calibrator,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        )

        return trt_mod



def quantize_models(models: list, input_shape: tuple, configuration: list, calibration_dataset, args): 
    quantized_models = []
    for i in range(len(configuration)):

        qmod = quantize_model(models[i], input_shape=input_shape, precision=configuration[i], calibration_dataset=calibration_dataset, args=args)
        sample_input = torch.randn((args.infer_batch_size,) + input_shape).cuda()
        input_shape = tuple(models[i](sample_input).cuda().shape)[1:]
        quantized_models.append(qmod)
        # computing the calibration dataset for next section
        calibration_data = []
        calibration_dataloader = DataLoader(calibration_dataset,
            batch_size=args.infer_batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,)
        
        for batch in calibration_dataloader:
            x, y = batch
            calibration_data.append(quantized_models[i](x).detach().cpu())
        calibration_data = torch.vstack(calibration_data)
        calibration_dataset = TensorDataset(calibration_data, torch.arange(calibration_data.shape[0]))
    return quantized_models


################################################# Loading Model #########################################################
args = parser.parse_arguments()
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.train_batch_size=2
args.infer_batch_size=1
args.fc_output_dim=512
args.resume = "/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_512/best_model.pth"

model = network.GeoLocalizationNet(args).cuda().eval()
model = util.resume_model(args, model)
models = split_model(model)

############################################## Benchmark Full Precision Performance ####################################
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
calibration_dataset = Subset(
    test_ds,
    list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)),
)

configuration = ["fp32", "fp32", "fp32", "fp32"]

sample_input = torch.randn((args.infer_batch_size,) + (3, 480, 640)).cuda()
st = time.time()
qmodels = quantize_models(models, input_shape=(3, 480, 640), args=args, configuration=configuration, calibration_dataset=calibration_dataset)
et = time.time()
hybridmodel = HybridModel(qmodels).cuda()
model.eval()
hybridmodel = torch.jit.trace(hybridmodel, sample_input)

try: 
    out = hybridmodel(sample_input.cuda()).detach().cpu()
except: 
    raise TypeError


recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, hybridmodel, args.test_method, None
)

recalls = recalls[0]
latency = util.benchmark_latency(model, input=sample_input)
print("latency: ", latency)
memory = measure_memory(hybridmodel)

print("recalls: ", recalls)

print("memory: ", memory)
print("quantize_time", et-st)
raise Exception


benchmark_recall = recalls
benchmark_latency = latency
benchmark_memory = memory

IMPORTANCE_WEIGHTS = [1., 0.9, 0.0] # recall, latency, memory
############################################# Create Objective Function ###############################################

def objective(params):
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    calibration_dataset = Subset(
        test_ds,
        list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)),
    )
    config = [params["block0_precision"], params["block1_precision"], params["block2_precision"], params["block3_precision"]]
    print("=============== ", config)
    qmodels = quantize_models(models, input_shape=(3, 480, 640), args=args, configuration=config, calibration_dataset=calibration_dataset)
    hybridmodel = HybridModel(qmodels).cuda()
    hybridmodel = torch.jit.trace(hybridmodel, sample_input)

    recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, hybridmodel, args.test_method, None
    )

    latency = util.benchmark_latency(model, input=sample_input)[0]

    memory = util.measure_memory(hybridmodel, (1, 3, 480, 640), args)

    memory_change = ((memory/benchmark_memory) - 1) * 100
    latency_change = ((latency/benchmark_latency) - 1) * 100
    recall_change = ((recalls[0]/benchmark_recall) - 1) * 100

    print(recall_change, latency_change, memory_change)
    loss = IMPORTANCE_WEIGHTS[2] * memory_change + IMPORTANCE_WEIGHTS[1]* latency_change - IMPORTANCE_WEIGHTS[0] * recall_change
    return {
        'loss': loss,
        'status': STATUS_OK
    }


############################################ Run Optimization ########################################################
space = {
    "block0_precision": hp.choice('block0_precision', ['fp32', 'fp16', 'int8']),
    "block1_precision": hp.choice('block1_precision', ['fp32', 'fp16', 'int8']),
    "block2_precision": hp.choice('block2_precision', ['fp32', 'fp16', 'int8']),
    "block3_precision": hp.choice('block3_precision', ['fp32', 'fp16', 'int8']),
}


best = fmin(
    fn=objective, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=5
)

b0 = ['fp32', 'fp16', 'int8'][best['block0_precision']]
b1 = ['fp32', 'fp16', 'int8'][best['block0_precision']]
b2 = ['fp32', 'fp16', 'int8'][best['block0_precision']]
b3 = ['fp32', 'fp16', 'int8'][best['block0_precision']]

best_config = [b0, b1, b2, b3]

print("===============================")
print("============= ", best_config)
print("===============================")

########################################### Store Best Configuration #################################################