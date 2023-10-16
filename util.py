
import re
import torch
import shutil
import logging
import torchscan
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
from torchvision import models
import os 
import torch_tensorrt
from torch.cuda.amp import autocast
import datasets_ws
import torch.nn as nn
from tqdm import tqdm
import copy

def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})

    for i, layer_name in enumerate(list(state_dict.keys())):
        if layer_name != list(model.state_dict().keys())[i]:
            print(layer_name != list(model.state_dict().keys())[i])
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca


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



def fuse_resnet18(fused_model):
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


def fuse_resnet50(fused_model):
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"],
                                                              ["conv3", "bn3"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)


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
                torch.quantization.fuse_modules(model, [name, bn_name], inplace=True)

        # Recursively apply to all child modules
        fuse_effcientnet(module)


def fuse_mobilenet(model):
    for name, module in model.named_children():
        if 'conv' in name:
            if isinstance(module, torch.nn.Conv2d):
                # fuse Conv2d + BatchNorm2d + ReLU
                torch.quantization.fuse_modules(model, [name, name[:name.rfind('.')] + '.norm'], inplace=True)
        # Recursively apply to child modules
        fuse_mobilenet(module)


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
    


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1.max(), y1.min())
            print(y2.max(), y2.min())
            return False
    return True




def prepareQAT(input_model, precision="fp16", backend="fbgemm", args=None):
    input_model.eval()

    # Fuse the Modules together
    if args.backbone.startswith("resnet18"):
        fused_model = copy.deepcopy(input_model)
        fuse_resnet18(fused_model)
        input_model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=input_model, model_2=fused_model, device="cpu", rtol=1e-01, atol=1e-04, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
        fused_model.train()
    elif args.backbone.startswith("resnet50"):
        fused_model = copy.deepcopy(input_model)
        fuse_resnet50(fused_model)
        input_model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=input_model, model_2=fused_model, device="cpu", rtol=1e-01, atol=1e-04, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
        fused_model.train()
    elif args.backbone.startswith("efficientnet"):
        fused_model = copy.deepcopy(input_model)
        fuse_effcientnet(fused_model)
        input_model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=input_model, model_2=fused_model, device="cpu", rtol=1e-01, atol=1e-04, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
        fused_model.train()
    elif args.backbone.startswith("shufflenet"):
        fused_model = copy.deepcopy(input_model)
        fuse_shufflenet(fused_model)
        input_model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=input_model, model_2=fused_model, device="cpu", rtol=1e-01, atol=1e-04, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
        fused_model.train()
    elif args.backbone.startswith("mobilenet"):
        fused_model = copy.deepcopy(input_model)
        fuse_mobilenet(fused_model)
        input_model.eval()
        fused_model.eval()
        assert model_equivalence(model_1=input_model, model_2=fused_model, device="cpu", rtol=1e-01, atol=1e-04, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"
        fused_model.train()

    # quantize the model to lower precision
    if precision == "fp16" or precision == "mixed":
        fused_model.half()
        fused_model.train()
        return fused_model
    elif precision == "fp32" or precision == "fp32_comp":
        fused_model.train()
        return fused_model
    elif precision == "int8_comp" or precision == "int8":
        qmodel = QuantizedModel(fused_model)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        qmodel.qconfig = quantization_config
        torch.quantization.prepare_qat(qmodel, inplace=True)
        qmodel.train()
        return qmodel



class mixedPrecision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with autocast():
            output = self.model(x)
        return output


def quantize_model(model, precision="fp16", calibration_dataset = None, args=None):
    if precision == "fp32":
        return model
    elif precision=="mixed":
        model = mixedPrecision(model)
        return model
    elif precision == "fp32_comp":
        trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((args.infer_batch_size, 3, 480, 640), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 22
        )
        return trt_model_fp32
    elif precision == "fp16_comp":
        trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((args.infer_batch_size, 3, 480, 640), dtype=torch.float32)],
            enabled_precisions = {torch.half}, # Run with FP16
            workspace_size = 1 << 22
        )

        return trt_model_fp16

    elif precision=="int8_comp":
        if calibration_dataset is None:
            raise Exception("must provide calibration dataset for int8 quantization")


        testing_dataloader = torch.utils.data.DataLoader(
            calibration_dataset, batch_size=args.infer_batch_size, shuffle=False, num_workers=1, drop_last=True
        )

        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((args.infer_batch_size, 3, 480, 640))],
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
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def measure_memory(model, input_size, args):
    if "comp" in args.precision:
        input = torch.randn(input_size, dtype=torch.float32).cuda()
        if args.precision == "mixed" or args.precision == "fp16":
            input = input.half()
        
        torch.jit.save(model, 'tmp_model.pt')
        size_in_bytes = os.path.getsize('tmp_model.pt')
        os.remove('tmp_model.pt')
        return size_in_bytes
    else: 
        torch.save(model.state_dict(), 'tmp_model.pt')
        size_in_bytes = os.path.getsize('tmp_model.pt')
        os.remove('tmp_model.pt')
        return size_in_bytes
    

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


    if precision == "fp16" or precision == "mixed":
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
