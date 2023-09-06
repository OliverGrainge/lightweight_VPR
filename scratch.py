import torch
import numpy as np
from torchvision.models import resnet50, resnet34, resnet18
from tqdm import tqdm


def benchmark_latency(
    model, input=None, input_size=(3, 224, 224), batch_size=64, repetitions=100, precision="fp32"
):
    if input is None:
        input_tensor = torch.randn((batch_size,) + input_size, dtype=torch.float32).to(
            "cuda"
        )
    
    if input is None:
        input_tensor = torch.randn((batch_size,) + input_size, dtype=torch.float32).cuda()
    else:
        input_tensor = input.cuda()


    if precision == "fp16":
        input_tensor = input_tensor.half().cuda()
    model.to("cuda")
    for _ in range(10):
        output = model(input_tensor)

    # Time measurement using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in tqdm(range(repetitions), desc="benchmarking latency"):
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
    if precision == "fp16":
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

    return max_batch_size - delta


import torch
import torchvision.models as models


def quantize_model(model, precision="fp16"):
    model.eval()
    if precision=="fp32":
        data = torch.randn(250, 3, 224, 224)
        labels = torch.randint(0, 10, size=(250,))
        testing_dataset = TensorDataset(data, labels)

        testing_dataloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=64, shuffle=False, num_workers=1
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(min_shape=(1, 3, 224, 224),
                                                        opt_shape=(64, 3, 224, 224),
                                                        max_shape=(200, 3, 224, 224),
                                                        dtype=torch.float)],
                                            enabled_precisions={torch.float32, torch.half, torch.int8},
                                            calibrator=calibrator,
                                            device={
                                                "device_type": torch_tensorrt.DeviceType.GPU,
                                                "gpu_id": 0,
                                                "dla_core": 0,
                                                "allow_gpu_fallback": False,
                                                "disable_tf32": False
                                            })
        return trt_mod
    elif precision=="fp16":
        data = torch.randn(250, 3, 224, 224)
        labels = torch.randint(0, 10, size=(250,))
        testing_dataset = TensorDataset(data, labels)

        testing_dataloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=64, shuffle=False, num_workers=1
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(min_shape=(1, 3, 224, 224),
                                                        opt_shape=(64, 3, 224, 224),
                                                        max_shape=(200, 3, 224, 224),
                                                        dtype=torch.float)],
                                            enabled_precisions={torch.half, torch.int8},
                                            calibrator=calibrator,
                                            device={
                                                "device_type": torch_tensorrt.DeviceType.GPU,
                                                "gpu_id": 0,
                                                "dla_core": 0,
                                                "allow_gpu_fallback": False,
                                                "disable_tf32": False
                                            })
        return trt_mod

    elif precision=="int8":
        data = torch.randn(250, 3, 224, 224)
        labels = torch.randint(0, 10, size=(250,))
        testing_dataset = TensorDataset(data, labels)

        testing_dataloader = torch.utils.data.DataLoader(
            testing_dataset, batch_size=64, shuffle=False, num_workers=1
        )
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(min_shape=(1, 3, 224, 224),
                                                        opt_shape=(64, 3, 224, 224),
                                                        max_shape=(200, 3, 224, 224),
                                                        dtype=torch.float)],
                                            enabled_precisions={torch.int8},
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


def prepareQAT(model, precision="fp16", backend="fbgemm", fuse_modules=None):
    if precision == "fp16":
        return model.half()
    elif precision == "fp32":
        return model
    elif precision == "int8":
        for modules in fuse_modules:
            torch.quantization.fuse_modules(model, fuse_modules, inplace=True)
        
        model = nn.Sequential(torch.quantization.QuantStub(),
                              *model,
                              torch.quantization.DeQuantStub())

        raise NotImplementedError


#import torch_tensorrt as torchtrt
import torch_tensorrt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn


model = nn.Sequential(
     nn.Conv2d(3,64,8),
     nn.ReLU(),
     nn.Conv2d(64, 128, 8),
     nn.ReLU()
)

model = models.resnet50().eval()

model = quantize_model(model, precision="int8")

print(benchmark_latency(model))