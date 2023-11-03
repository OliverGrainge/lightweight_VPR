import parser
import test
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#import datasets_ws
import util
from model import network

args = parser.parse_arguments()
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.train_batch_size = 2
args.infer_batch_size = 1
args.fc_output_dim = 1024
args.device = "cpu"
args.dataset_name = "eynsham"
args.datasets_folder = "/Users/olivergrainge/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.resume = "/Users/olivergrainge/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth"


model = network.GeoLocalizationNet(args).eval()

################################### Quantization Functions ###################################


def quantize_tensor(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_tensor = (tensor / scale).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale
    return q_tensor


def quantize_channel(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_channels = []
    reshaped_tensor = tensor.permute(1, 0, 2, 3)
    for i in range(reshaped_tensor.size(0)):
        channel_weights = reshaped_tensor[i]
        q_channel = (channel_weights / scale[i]).clamp(qmin, qmax).round()
        q_channel = q_channel * scale[i]
        q_channels.append(q_channel)
    q_channels = torch.stack(q_channels).permute(1, 0, 2, 3)
    return q_channels


def quantize_filter(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_tensors = []
    for i, channel in enumerate(tensor):
        q_tensor = (channel / scale[i]).clamp(qmin, qmax).round()
        q_tensor = q_tensor * scale[i]
        q_tensors.append(q_tensor)
    q_tensor = torch.stack(q_tensors)
    return q_tensor


def get_qrange(precision: str = "int8"):
    if precision == "fp32":
        return None, None
    elif precision == "fp16":
        return None, None
    elif precision == "int32":
        qmin = -(2.0 ** (32 - 1))  # Minimum quantization value
        qmax = 2.0 ** (32 - 1) - 1  # Maximum quantization value
    elif precision == "int8":
        qmin = -(2.0 ** (8 - 1))
        qmax = 2.0 ** (8 - 1) - 1
    elif precision == "int4":
        qmin = -(2.0 ** (4 - 1))
        qmax = 2.0 ** (4 - 1) - 1
    else:
        raise Exception("Can only quantize to int4, int8, int32, fp16, fp32 precisions")
    return torch.tensor(qmin, requires_grad=True), torch.tensor(
        qmax, requires_grad=True
    )


def quantize_weights(
    tensor: torch.tensor,
    scale: torch.tensor,
    precision: str = "int8",
    granularity="tensor",
):
    # Floating Point quantization
    if precision == "fp32":
        return tensor
    elif precision == "fp16":
        return tensor.half().float()

    qmin, qmax = get_qrange(precision=precision)
    # Integer Quantization
    if granularity == "tensor" or tensor.ndim <= 2 and precision:
        tensor = quantize_tensor(tensor, scale, qmin, qmax)
    elif granularity == "channel":
        tensor = quantize_channel(tensor, scale, qmin, qmax)
    elif granularity == "filter":
        tensor = quantize_filter(tensor, scale, qmin, qmax)
    return tensor


########################################## Calibrators ##############################################


class ActivationCollector:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        self.activations.append(output)

    def register_hooks(self, model, layers=(nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        hooks = []
        for layer in model.children():
            if isinstance(layer, layers):
                hooks.append(layer.register_forward_hook(self.hook_fn))
            self.register_hooks(layer)  # Recursive call for nested layers
        return hooks

    def __call__(self, model, input):
        self.activations = []  # Reset previous activations
        hooks = self.register_hooks(model)
        model(input)
        for hook in hooks:
            hook.remove()
        return self.activations


def minmax_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None
    if granularity == "tensor" or tensor.ndim <= 2:
        qmin, qmax = get_qrange(precision=precision)
        min_val, max_val = tensor.min(), tensor.max()
        max_abs = max(abs(min_val), abs(max_val))
        scale = max_abs / max(qmax, -qmin)
        return scale.clone().detach().requires_grad_(True)

    elif granularity == "channel":
        scales = []
        reshaped_tensor = (
            tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
        )
        for i in range(reshaped_tensor.size(0)):
            channel = reshaped_tensor[i]
            qmin, qmax = get_qrange(precision=precision)
            min_val, max_val = channel.min(), channel.max()
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / max(qmax, -qmin)
            scales.append(scale)
        return torch.tensor(scales).clone().detach().requires_grad_(True)

    elif granularity == "filter":
        scales = []
        for filter in tensor:
            qmin, qmax = get_qrange(precision=precision)
            min_val, max_val = filter.min(), filter.max()
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / max(qmax, -qmin)
            scales.append(scale)
        scales = torch.tensor(scales).clone().detach().requires_grad_(True)
        return scales


def kl_divergence(p, q):
    # KL divergence between two distributions
    return (p * (p / q).log()).sum()


def entropy_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None

    tensor.requires_grad_(True)
    # Calculate the histogram of the activations
    # The number of bins could be set to the range of the precision
    qmin, qmax = get_qrange(precision=precision)
    num_bins = int(qmax - qmin + 1)

    if granularity == "tensor" or tensor.ndim <= 2:
        hist = torch.histc(
            tensor, bins=num_bins, min=tensor.min().item(), max=tensor.max().item()
        )
        prob_dist = hist / hist.sum()
    elif granularity == "filter":
        hist = torch.stack(
            [
                torch.histc(
                    filt, bins=num_bins, min=filt.min().item(), max=filt.max().item()
                )
                for filt in tensor
            ]
        )
    elif granularity == "channel":
        tensor_reshaped = (
            tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
        )
        hist = torch.stack(
            [
                torch.histc(
                    chan, bins=num_bins, min=chan.min().item(), max=chan.max().item()
                )
                for chan in tensor_reshaped
            ]
        )

    # Initialize the scale with some reasonable values
    if granularity == "tensor" or tensor.ndim <= 2:
        max_abs = max(abs(tensor.min()), abs(tensor.max()))
        scale = torch.tensor(max_abs / max(qmax, -qmin), requires_grad=True)
        scale = torch.nn.Parameter(scale)
    elif granularity == "filter":
        max_abs = torch.amax(tensor.abs(), dim=(1, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)
    elif granularity == "channel":
        max_abs = torch.amax(tensor.abs(), dim=(0, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)

    optimizer = Adam([scale], lr=0.01)

    for _ in range(1000):  # Run for a number of iterations
        optimizer.zero_grad()

        # Compute the quantized tensor
        if granularity == "tensor" or tensor.ndim <= 2:
            quantized_tensor = quantize_tensor(tensor, scale, qmin, qmax)
            print(quantized_tensor.requires_grad)

            # Compute the histogram of the quantized tensor
            q_hist = torch.histc(
                quantized_tensor,
                bins=num_bins,
                min=tensor.min().item(),
                max=tensor.max().item(),
            )
            q_prob_dist = q_hist / q_hist.sum()

            # Compute the KL divergence
            loss = kl_divergence(prob_dist, q_prob_dist)
            print(loss.requires_grad)

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        elif granularity == "filter":
            quantized_tensor = quantize_filter(tensor, scale, qmin, qmax)

            # Compute the histogram of the quantized tensor
            q_hist = torch.stack(
                [
                    torch.histc(
                        filt, bins=num_bins, min=tensor[i].min(), max=tensor[i].max()
                    )
                    for i, filt in enumerate(quantized_tensor)
                ]
            )
            q_prob_dist = q_hist / q_hist.sum(1, keepdims=True)

            # Compute the KL divergence
            loss = torch.stack(
                [
                    kl_divergence(prob_dist[i], q_prob_dist[i])
                    for i in range(q_hist.size(0))
                ]
            ).mean()

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        elif granularity == "channel":
            quantized_tensor = quantize_channel(tensor, scale, qmin, qmax)

            # Compute the histogram of the quantized tensor
            reshaped_qtensor = (
                quantized_tensor.permute(1, 0, 2, 3)
                .contiguous()
                .view(quantized_tensor.size(1), -1)
            )
            q_hist = torch.stack(
                [
                    torch.histc(
                        chan,
                        bins=num_bins,
                        min=tensor_reshaped[i].min(),
                        max=tensor_reshaped[i].max(),
                    )
                    for i, chan in enumerate(reshaped_qtensor)
                ]
            )
            q_prob_dist = q_hist / q_hist.sum(1, keepdims=True)

            # Compute the KL divergence
            loss = torch.stack(
                [
                    kl_divergence(prob_dist[i], q_prob_dist[i])
                    for i in range(q_hist.size(0))
                ]
            ).mean()

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        # Ensure that scale is always positive and zero_point is within range
        scale.data.clamp_(min=1e-8)
    print("THEERE")
    return scale.item()


def mse_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None

    tensor.requires_grad_(True)
    # Calculate the histogram of the activations
    # The number of bins could be set to the range of the precision
    qmin, qmax = get_qrange(precision=precision)
    # Initialize the scale with some reasonable values
    if granularity == "tensor" or tensor.ndim <= 2:
        max_abs = max(abs(tensor.min()), abs(tensor.max()))
        scale = torch.tensor(max_abs / max(qmax, -qmin))
        scale = torch.nn.Parameter(scale)
    elif granularity == "filter":
        max_abs = torch.amax(tensor.abs(), dim=(1, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)
    elif granularity == "channel":
        max_abs = torch.amax(tensor.abs(), dim=(0, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)

    optimizer = Adam([scale], lr=0.00001)
    for round in range(20):  # Run for a number of iterations
        optimizer.zero_grad()

        # Compute the quantized tensor
        if granularity == "tensor" or tensor.ndim <= 2:
            quantized_tensor = quantize_tensor(tensor, scale, qmin, qmax)

        elif granularity == "filter":
            quantized_tensor = quantize_filter(tensor, scale, qmin, qmax)

        elif granularity == "channel":
            quantized_tensor = quantize_channel(tensor, scale, qmin, qmax)

        # Ensure that scale is always positive and zero_point is within range
        loss = F.mse_loss(tensor, quantized_tensor)

        # Perform the optimization step
        loss.backward()
        optimizer.step()
        scale.data.clamp_(min=1e-8)
    return scale.detach()

########################################## Activation Qauntization ##############################################

# This will be run after every relu function through a hook function 

def activation_quantization(module, input, output, precision="fp16", granularity="tensor"):
    scale = minmax_scaler(output, precision=precision, granularity=granularity)
    qoutput = quantize_weights(output, scale, precision=precision, granularity=granularity) 
    return qoutput


######################################## RUN QUANTIZATION ###############################################

new_state_dict = {}


def quantize_recursive(
    module,
    module_name,
    state_dict,
    configuration,
    granularity,
    calibration="minmax",
    new_state_dict=None,
    idx=None,
):
    if new_state_dict is None:
        new_state_dict = {}
    if idx is None:
        idx = 0

    is_leaf_module = len(list(module.children())) == 0
    if is_leaf_module and isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
        # Apply the function to the module's weights and bias if they exist
        for name, param in module.named_parameters():
            if name == "weight":
                param_key = f"{module_name}.{name}" if module_name else name

                weight = param.data.clone().detach()
                if calibration == "minmax":
                    scale = minmax_scaler(
                        weight, precision=configuration[idx], granularity=granularity
                    )
                elif calibration == "entropy":
                    scale = entropy_scaler(
                        weight, precision=configuration[idx], granularity=granularity
                    )
                elif calibration == "mse":
                    print(idx)
                    scale = mse_scaler(
                        weight, precision=configuration[idx], granularity=granularity
                    )
                qtensor = quantize_weights(
                    weight, scale, precision=configuration[idx], granularity=granularity
                )
                new_state_dict[param_key] = qtensor
        idx += 1
    else:
        # Call this function recursively for each submodule
        for name, submodule in module.named_children():
            if name:  # avoid self-recursion on the module itself
                full_name = f"{module_name}.{name}" if module_name else name
                new_state_dict, idx = quantize_recursive(
                    submodule,
                    full_name,
                    state_dict,
                    configuration,
                    granularity,
                    calibration=calibration,
                    new_state_dict=new_state_dict,
                    idx=idx,
                )
    return new_state_dict, idx


def quantize_model(model, configuration, granularity="tensor", calibration="minmax"):
    state_dict = model.state_dict()
    new_state_dict, idx = quantize_recursive(
        model, None, state_dict, configuration, granularity, calibration=calibration
    )
    for key in state_dict.keys():
        if key not in list(new_state_dict.keys()):
            new_state_dict[key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    return model


############################################# QUANTIZATION ###############################################
model = network.GeoLocalizationNet(args).eval()
qmodel = network.GeoLocalizationNet(args).eval()
model = util.resume_model(args, model)
qmodel = util.resume_model(args, qmodel)


configuration = np.random.choice(["int8"], size=103)

qmodel = quantize_model(qmodel, configuration, granularity="channel", calibration="minmax")


def check_equivalence(model1, model2):
    sd2 = model2.state_dict()
    sd1 = model1.state_dict()
    for key in sd1:
        print(key, (sd2[key].detach().numpy() == sd1[key].detach().numpy()).all())


check_equivalence(model, qmodel)




########################################## Model Evaluation ###########################################

"""
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")

recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, model, args.test_method, None
)
"""


