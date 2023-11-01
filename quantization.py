import torch 
import torchvision
from model import network
import parser
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import numpy as np

args = parser.parse_arguments()
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.train_batch_size=2
args.infer_batch_size=1
args.fc_output_dim=1024
args.resume = "/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth"


RANGE = [-20, 20]
BINS = 500


def flatten_module_recursive(module):
    layers = []
    if len(list(module.children())) == 0:  # If it's a leaf node, just append it.
        return [module]
    for child in module.children():
        layers.extend(flatten_module_recursive(child))
    return nn.Sequential(*layers)





def minmax_scaler(tensor, nbits):
    qmin = -2.**(nbits - 1)  # Minimum quantization value
    qmax = 2.**(nbits - 1) - 1  # Maximum quantization value
    min_val, max_val = tensor.min(), tensor.max()
    max_abs = max(abs(min_val), abs(max_val))
    scale = max_abs / max(qmax, -qmin)
    return scale, qmin, qmax



def qconv_perfilter(tensor, nbits=8, calibration_data=None):
    if nbits == 16:
        tensor = tensor.half().float()
        return tensor
    elif nbits == 32:
        return tensor

    scales = []
    q_tensors = []

    for channel in range(tensor.size(0)):
        if calibration_data == None:
            scale, qmin, qmax = minmax_scaler(tensor[channel], nbits=nbits)
        else: 
            raise NotImplementedError
        scales.append(scale)
        q_tensor = (tensor[channel] / scale).clamp(qmin, qmax).round()
        q_tensor = q_tensor * scale 
        q_tensors.append(q_tensor)

    q_tensor = torch.stack(q_tensors)
    return q_tensor

def qconv_perchannel(tensor, nbits=8, calibration_data=None):
    if nbits == 16:
        tensor = tensor.half().float()
        return tensor
    elif nbits == 32:
        return tensor

    qmin = 0.
    qmax = 2.**nbits - 1.
    q_channels = []

    # Reshape tensor such that each input channel is a separate row
    reshaped_tensor = tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)

    # Loop through each reshaped input channel
    for i in range(reshaped_tensor.size(0)):
        channel_weights = reshaped_tensor[i]
        if calibration_data == None:
            scale, qmin, qmax = minmax_scaler(channel_weights, nbits=nbits)
        else: 
            raise NotImplementedError

        q_channel = (channel_weights / scale).clamp(qmin, qmax).round()
        q_channel = q_channel * scale 
        q_channels.append(q_channel)

    # Reshape the quantized channels back to original shape
    q_channels = torch.stack(q_channels).view_as(tensor.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    return q_channels

def qconv_pertensor(tensor, nbits=8, calibration_data=None):
    if nbits == 16:
        tensor = tensor.half().float()
        return tensor
    elif nbits == 32:
        return tensor

    qmin = 0.
    qmax = 2.**nbits - 1.
    if calibration_data == None:
        scale, qmin, qmax = minmax_scaler(tensor, nbits=nbits)
    else: 
        raise NotImplementedError

    if calibration_data == None:
        scale, qmin, qmax = minmax_scaler(tensor, nbits=nbits)
    q_tensor = (tensor / scale).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale 
    return q_tensor


def qlin_pertensor(tensor, nbits=8, calibration_data=None):
    if nbits == 16:
        tensor = tensor.half().float()
        return tensor
    elif nbits == 32:
        return tensor
    if calibration_data == None:
        scale, qmin, qmax = minmax_scaler(tensor, nbits=nbits)
    else: 
        raise NotImplementedError
    q_tensor = (tensor / scale ).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale
    return q_tensor


def quantize_layer(layer, nbits, calibration_data=None, granularity="tensor"):
    if isinstance(layer, nn.Conv2d):
        if granularity == "tensor":
            layer = qconv_pertensor(layer, nbits=nbits)
        elif granularity == "channel":
            layer = qconv_perchannel(layer, nbits=nbits)
        if granularity == "filter":
            layer = qconv_perfilter(layer, nbits=nbits)

        # Linear Layers
    elif isinstance(layer, nn.Linear):
        layer = qlin_pertensor(layer, nbits=nbits)

        # BatchNorm Lyaers 
    elif isinstance(layer, nn.BatchNorm2d):
        layer = qlin_pertensor(layer, nbits=nbits)

    return layer


def count_conv2d_layers_recursive(model, count=0):
    for module in model.children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
            count += 1
        elif isinstance(module, nn.Module):
            # If it's a submodule (e.g., Sequential or another module),
            # recursively count Conv2d layers within the submodule
            count = count_conv2d_layers_recursive(module, count)

    return count

def quantize_network(model, configuration, granularity="tensor"):
    i = 0
    qdict = {}
    for key, tensor in model.state_dict().items():
        # Convolutional Lyaers
        if "conv" in key and "0.weight" in key or "conv.2.weight" in key:
            if granularity == "tensor":
                tensor = qconv_pertensor(tensor, nbits=configuration[i])
                qdict[key] = tensor
                i += 1
            elif granularity == "channel":
                tensor = qconv_perchannel(tensor, nbits=configuration[i])
                qdict[key] = tensor
                i += 1
            if granularity == "filter":
                tensor = qconv_perfilter(tensor, nbits=configuration[i])
                qdict[key] = tensor
                i += 1

        # Batch Norm
        elif ("conv" in key and "1.weight" in key) or "conv.3.weight" in key:
            tensor = qlin_pertensor(tensor, nbits=configuration[i])
            qdict[key] = tensor
            i += 1
    
        elif "weight" in key and "aggregation" in key:
            tensor = qlin_pertensor(tensor, nbits=configuration[i])
            qdict[key] = tensor
            i += 1
    
        else: 
            qdict[key] = tensor

    model.load_state_dict(qdict)
    return model
############################################## Quantization ##########################################




model = network.GeoLocalizationNet(args).eval()

configuration = np.random.choice([6, 16, 32], size=count_conv2d_layers_recursive(model))


sample_input = torch.randn(1, 3, 480, 640)

out1 = model(sample_input).flatten().detach().numpy()

qmodel = quantize_network(model, configuration)

out2 = qmodel(sample_input).flatten().detach().numpy()

for i in range(len(out1)):
    print(out1[i], out2[i])





        







