import torch
import numpy as np
from torchvision.models import resnet50, resnet34, resnet18
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.cuda.amp import autocast
from torch import nn
import logging
import torch_tensorrt
import torch.nn.functional as F
import torchvision
from torchvision import transforms


import os
os.environ["CUDA_LAZY_DEBUG"] = "1"


def benchmark_latency(
    model, input=None, input_size=(3, 480, 640), batch_size=1, repetitions=100, precision="fp32"
):

    logging.debug("Testing Model Latency")
    if input is None:
        input_tensor = torch.randn((batch_size,) + input_size, dtype=torch.float32).to(
            "cuda"
        )
    
    if input is None:
        input_tensor = torch.randn((batch_size,) + input_size, dtype=torch.float32).cuda()
    else:
        input_tensor = input.cuda()


    if precision == "fp16" or precision == "mixed" or precision == "fp16_comp":
        input_tensor = input_tensor.half().cuda()
    model.to("cuda")
    for _ in range(10):
        output = model(input_tensor)
        del output

    # Time measurement using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in tqdm(range(repetitions)):
        start_event.record()
        output = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    average_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return average_latency, std_latency


def benchmark_throughput(model, input_shape=(3, 224, 224), precision="fp32"):
    # Make sure model is on GPU
    model = model.cuda()

    # Create dummy data
    batch_size = find_max_batch_size(model, input_shape=input_shape, precision=precision, device="cuda")
    torch.cuda.empty_cache()
    input_tensor = torch.randn(batch_size, 3, 224, 224).float().cuda()
    if precision == "fp16" or precision == "mixed" or precision == "fp16_comp":
        input_tensor = input_tensor.half().cuda()

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            output = model(input_tensor)

    # Measure throughput using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    num_batches = 100
    for _ in tqdm(range(num_batches), desc="benchmarking latency"):
        output = model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    elapsed_time = elapsed_time / 1000  # Convert to seconds

    throughput = (batch_size * num_batches) / elapsed_time
    return throughput


def find_max_batch_size(model, input_shape, device, delta=5, precision='fp32'):
    max_batch_size = 1
    model = model.to(device)

    while True:
        try:
            # Create a random tensor with the specified batch size and input shape
            dummy_input = torch.randn([max_batch_size] + list(input_shape)).to(device)

            if precision == 'fp16':
                dummy_input = dummy_input.half().cuda()

            # Perform forward pass
            dummy_output = model(dummy_input)

            # Clear cache to make sure memory is actually available
            del dummy_output
            torch.cuda.empty_cache()
            

            # Double the batch size for the next iteration
            max_batch_size += delta

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    f"Batch size of {max_batch_size} failed. Max feasible batch size is {max_batch_size - delta}."
                )
                break
            else:
                raise e

    return int((max_batch_size - delta) * 0.5)


import torch
import torchvision.models as models

def check_nan_inf(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in {name}')
        if torch.isinf(param).any():
            print(f'Inf detected in {name}')



class mixedPrecision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with autocast():
            output = self.model(x)
        return output

def quantize_model(model, precision="fp16"):
    if precision == "fp32":
        return model
    elif precision=="mixed":
        model = mixedPrecision(model)
        return model
    elif precision=="fp16":
        model = model.half()
        return model
    elif precision == "fp32_comp":
        trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 22
        )
        return trt_model_fp32
    elif precision == "fp16_comp":
        trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
            enabled_precisions = {torch.half}, # Run with FP16
            workspace_size = 1 << 22
        )

        return trt_model_fp16

    elif precision=="int8_comp":
        testing_dataset = torchvision.datasets.CIFAR10(
                        root="./data",
                        train=False,
                        download=True,
                        transform=transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]
                        ),
                    )

        testing_dataloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
                                            enabled_precisions={torch.float, torch.half, torch.int8},
                                            calibrator=calibrator,
                                            device={
                                                "device_type": torch_tensorrt.DeviceType.GPU,
                                                "gpu_id": 0,
                                                "dla_core": 0,
                                                "allow_gpu_fallback": False,
                                                "disable_tf32": False
                                            })
        
        return trt_mod

    elif precision=="int4":
        raise NotImplementedError


################################## QAT Training #####################################

class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def prepareQAT(model, precision="fp16", backend="fbgemm", fuse_modules=None):
    if precision == "fp16_comp" or precision == "fp16" or precision == "mixed":
        model.half()
        model.train()
        return model
    elif precision == "fp32":
        model.train()
        return model
    elif precision == "int8_comp":
        qmodel = QuantizedModel(model)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        qmodel.qconfig = quantization_config
        torch.quantization.prepare_qat(qmodel, inplace=True)
        qmodel.train()
        return qmodel

        
        

from model.network import GeoLocalizationNet


import math 
from torch.utils.data import DataLoader, SubsetRandomSampler




class ShuffleNetV2Features(nn.Module):
    def __init__(self, original_model):
        super(ShuffleNetV2Features, self).__init__()
        self.features = original_model.conv1
        self.maxpool = original_model.maxpool
        self.stage2 = original_model.stage2
        self.stage3 = original_model.stage3
        self.stage4 = original_model.stage4
    
    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x




def fuse_resnet18(model):
    model.eval()

    fuse_list = [['conv1', 'bn1', 'relu'],
             ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],
             ['layer1.0.conv2', 'layer1.0.bn2'],
             ['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu'],
             ['layer1.1.conv2', 'layer1.1.bn2']]
    
    for fuse_modules in fuse_list:
        torch.quantization.fuse_modules(model, fuse_modules)

def fuse_resnet50(model):
    model.eval()
    # Fuse the first Conv2d + BatchNorm2d + ReLU layers
    torch.quantization.fuse_modules(model, ['conv1', 'bn1', 'relu'])
    # Fuse Conv2d + BatchNorm2d + ReLU layers for each Bottleneck block
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for bottleneck in layer:
            # Fuse the layers in the Bottleneck sub-block
            torch.quantization.fuse_modules(bottleneck, ['conv1', 'bn1', 'relu'])
            torch.quantization.fuse_modules(bottleneck, ['conv2', 'bn2', 'relu'])
            torch.quantization.fuse_modules(bottleneck, ['conv3', 'bn3'])


def fuse_shufflenet(model):
    model.eval()

    # Fuse Conv2d + BatchNorm2d for the first Conv layer
    torch.quantization.fuse_modules(model, ['conv1.0', 'conv1.1'])

    # Fuse the Conv2d + BatchNorm2d + ReLU layers in each ShuffleNetUnit
    for stage in [model.stage2, model.stage3, model.stage4]:
        for i, module in enumerate(stage.modules()):
            if isinstance(module, models.shufflenetv2.InvertedResidual):
                for j, name in enumerate(['branch1.0', 'branch1.1']):
                    torch.quantization.fuse_modules(module, [f'branch1.{j}.0', f'branch1.{j}.1'])
                for j, name in enumerate(['branch2.0', 'branch2.1', 'branch2.3', 'branch2.4']):
                    torch.quantization.fuse_modules(module, [f'branch2.{j}.0', f'branch2.{j}.1'])

def fuse_effcientnet(model):
    model.eval()
    # Loop through the sub-modules to search for Conv2d + BatchNorm2d layers
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            # Get the name of the BatchNorm2d layer
            bn_name = name.replace("conv", "bn")
            
            # Check if the next named module is a BatchNorm2d layer
            next_name, next_module = next(model.named_children())
            if bn_name == next_name and isinstance(next_module, torch.nn.BatchNorm2d):
                # Fuse Conv2d + BatchNorm2d
                fuse_modules(model, [name, bn_name], inplace=True)

        # Recursively apply to all child modules
        fuse_effcientnet(module)




if __name__ == "__main__":
    import timm
    import torchvision.models as models
    model = timm.create_model("efficientnet_b0", pretrained=True)

    fuse_effcientnet(model)
    print(model)
