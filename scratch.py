import torch
import numpy as np
from torchvision.models import resnet50, resnet34, resnet18
from tqdm import tqdm


def benchmark_latency(
    model, input=None, input_size=(3, 224, 224), batch_size=(12,), repetitions=100
):
    if input is None:
        input_tensor = torch.randn(batch_size + input_size, dtype=torch.float32).to(
            "cuda"
        )
    else:
        input_tensor = input.cuda()
    model.to("cuda")
    for _ in range(10):
        output = model(input_tensor)

    # Time measurement using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(repetitions):
        start_event.record()
        output = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    average_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return average_latency, std_latency


def benchmark_throughput(model, input_shape=(3, 224, 224)):
    # Make sure model is on GPU
    model = model.cuda()

    # Create dummy data
    batch_size = find_max_batch_size(model, input_shape=input_shape, device="cuda")
    torch.cuda.empty_cache()
    input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            output = model(input_tensor)

    # Measure throughput using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    num_batches = 100
    for _ in range(num_batches):
        output = model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    elapsed_time = elapsed_time / 1000  # Convert to seconds

    throughput = (batch_size * num_batches) / elapsed_time
    return throughput


def find_max_batch_size(model, input_shape, device, delta=5):
    max_batch_size = 1
    model = model.to(device)

    while True:
        try:
            # Create a random tensor with the specified batch size and input shape
            dummy_input = torch.randn([max_batch_size] + list(input_shape)).to(device)

            # Perform forward pass
            dummy_output = model(dummy_input)

            print(f"Batch size of {max_batch_size} succeeded.")

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
from torch.ao.quantization import get_default_qconfig, fuse_modules, prepare, convert

# Step 1: Load and prepare the model
model = models.resnet18(pretrained=True).cuda()
model.eval()

# Step 2: Set the model to evaluation mode
# Already set using model.eval()

# Step 3: Fuse Conv and Batch Norm layers for optimization
# For resnet, layer fusion can be performed as below
model = torch.ao.quantization.fuse_modules(
    model,
    [
        ["conv1", "bn1"],
        ["layer1.0.conv1", "layer1.0.bn1"],
        ["layer1.0.conv2", "layer1.0.bn2"],
        # Add more layers as needed
    ],
)

# Step 4: Specify Quantization Config
# In this case, we use the default quantization settings
model.qconfig = get_default_qconfig("fbgemm")  # For x86 CPUs
# model.qconfig = get_default_qconfig("qnnpack")  # For ARM CPUs

# Step 5: Calibrate the Model
# 'prepare' makes the model ready for static quantization by inserting observers
model_prepared = prepare(model)

# Calibrate the prepared model to determine quantization parameters
# Use a subset of your data as the calibration dataset
sample_data = torch.randn(32, 3, 224, 224)
model_prepared(sample_data.cuda())

# Step 6: Convert to Quantized Model
# The actual quantization happens in this step
model_quantized = convert(model_prepared)
# Test the quantized model similarly as you would with `model`
# Now, you can use model_quantized for inference.
out = model(sample_data.cuda())
print(out.shape)
