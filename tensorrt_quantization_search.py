import io
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

import datasets_ws
import util
from model import network

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import parser
import pickle


def measure_memory(model):
    script0 = model.block0
    script1 = model.block1
    script2 = model.block2

    torch.jit.save(script0, "tmp0_model.pt")
    torch.jit.save(script1, "tmp1_model.pt")
    torch.jit.save(script2, "tmp2_model.pt")


    size_in_bytes = os.path.getsize("tmp0_model.pt")
    size_in_bytes += os.path.getsize("tmp1_model.pt")
    size_in_bytes += os.path.getsize("tmp2_model.pt")

    os.remove("tmp0_model.pt")
    os.remove("tmp1_model.pt")
    os.remove("tmp2_model.pt")

    return size_in_bytes

class PartitionedModel(nn.Module):
    def __init__(self, modules):
        super(PartitionedModel, self).__init__()
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

def split_model(model):
    # Assuming that the backbone has 13 InvertedResidual blocks as in the MobileNetV2 architecture
    backbone_partitions = [model.backbone[:15], model.backbone[15:16], model.backbone[16:], model.aggregation[:]]
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
            enabled_precisions={torch.half, torch.int8},
            calibrator=calibrator,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
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
args.fc_output_dim=1024
args.resume = "/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth"

model = network.GeoLocalizationNet(args).cuda().eval()
model = util.resume_model(args, model)
models = split_model(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





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

try: 
    out = hybridmodel(sample_input.cuda()).detach().cpu()
except: 
    raise TypeError

memory = measure_memory(hybridmodel)
print("memory: ", memory)

recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, hybridmodel, args.test_method, None
)
print("recalls: ", recalls)

recalls = recalls[0]
latency = util.benchmark_latency(model, input=sample_input)
print("latency: ", latency)


print("quantize_time", et-st)
"""

benchmark_recall = 69.26
benchmark_latency = 2.778
benchmark_memory = 12841612
sample_input = torch.randn((args.infer_batch_size,) + (3, 480, 640)).cuda()

IMPORTANCE_WEIGHTS = [0., 1.0, 0.16] # recall, latency, memory
############################################# Create Objective Function ###############################################
evaled_configs = []
evaled_recalls = []
evaled_latency = []
evaled_memory = []
evaled_losses = []
def objective(params):
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    calibration_dataset = Subset(
        test_ds,
        list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)),
    )
    config = [params["block0_precision"], params["block1_precision"], params["block2_precision"], 'fp32']

    evaled_configs.append(config)

    qmodels = quantize_models(models, input_shape=(3, 480, 640), args=args, configuration=config, calibration_dataset=calibration_dataset)
    hybridmodel = HybridModel(qmodels).cuda()

    recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, hybridmodel, args.test_method, None
    )
    evaled_recalls.append(recalls)

    memory = measure_memory(hybridmodel)
    evaled_memory.append(memory)

    latency = util.benchmark_latency(hybridmodel, input=sample_input)[0]
    evaled_latency.append(latency)

    memory_change = ((memory/benchmark_memory) - 1) * 100
    latency_change = ((latency/benchmark_latency) - 1) * 100
    recall_change = ((recalls[0]/benchmark_recall) - 1) * 100


    loss = IMPORTANCE_WEIGHTS[2] * memory_change + IMPORTANCE_WEIGHTS[1]* latency_change - IMPORTANCE_WEIGHTS[0] * recall_change
    evaled_losses.append(loss)

    with open('configs.pkl', 'wb') as f:
        pickle.dump(evaled_configs, f)

    with open('latency.pkl', 'wb') as f:
        pickle.dump(evaled_latency, f)

    with open('memory.pkl', 'wb') as f:
        pickle.dump(evaled_memory, f)

    with open('recall.pkl', 'wb') as f:
        pickle.dump(evaled_recalls, f)

    with open('loss.pkl', 'wb') as f:
        pickle.dump(loss, f)

    return {
        'loss': loss,
        'status': STATUS_OK
    }


############################################ Run Optimization ########################################################
space = {
    "block0_precision": hp.choice('block0_precision', ['fp32', 'fp16', 'int8']),
    "block1_precision": hp.choice('block1_precision', ['fp32', 'fp16', 'int8']),
    "block2_precision": hp.choice('block2_precision', ['fp32', 'fp16', 'int8']),
}


best = fmin(
    fn=objective, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=27
)

b0 = ['fp32', 'fp16', 'int8'][best['block0_precision']]
b1 = ['fp32', 'fp16', 'int8'][best['block1_precision']]
b2 = ['fp32', 'fp16', 'int8'][best['block2_precision']]

best_config = [b0, b1, b2, b3]

print("===============================")
print("============= ", best_config)
print("===============================")

########################################### Store Best Configuration #################################################
"""