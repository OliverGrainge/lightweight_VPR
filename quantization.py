import torch 
import torchvision
from model import network
import parser
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 

args = parser.parse_arguments()
args.backbone = "mobilenetv2conv4"
args.aggregation = "mac"
args.train_batch_size=2
args.infer_batch_size=1
args.fc_output_dim=1024
args.resume = "/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth"

model = network.GeoLocalizationNet(args).eval()


def flatten_module_recursive(module):
    layers = []
    if len(list(module.children())) == 0:  # If it's a leaf node, just append it.
        return [module]
    for child in module.children():
        layers.extend(flatten_module_recursive(child))
    return nn.Sequential(*layers)



model = flatten_module_recursive(model)

def minmax_scaler(tensor, nbits):
    qmin = 0.
    qmax = 2.**nbits - 1.
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    return scale, qmin, qmax
    

def qconv_perfilter(layer, nbits=8):
    tensor = layer.weight.data
    scales = []
    q_tensors = []

    for channel in range(tensor.size(0)):
        min_val, max_val = tensor[channel].min(), tensor[channel].max()
        scale, qmin, qmax = minmax_scaler(tensor[channel], nbits=nbits)
        scales.append(scale)
        q_tensor = (tensor[channel] / scale + 0.5).clamp(qmin, qmax).round()
        q_tensor = q_tensor * scale + min_val
        q_tensors.append(q_tensor)

    layer.weight.data = torch.stack(q_tensors)
    return layer

def qconv_perchannel(layer, nbits=8):
    tensor = layer.weight.data
    qmin = 0.
    qmax = 2.**nbits - 1.
    q_channels = []

    # Reshape tensor such that each input channel is a separate row
    reshaped_tensor = tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)

    # Loop through each reshaped input channel
    for i in range(reshaped_tensor.size(0)):
        channel_weights = reshaped_tensor[i]
        min_val, max_val = channel_weights.min(), channel_weights.max()
        scale = (max_val - min_val) / (qmax - qmin)
        
        q_channel = (channel_weights / scale + 0.5).clamp(qmin, qmax).round()
        q_channel = q_channel * scale + min_val
        q_channels.append(q_channel)

    # Reshape the quantized channels back to original shape
    q_channels = torch.stack(q_channels).view_as(tensor.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    layer.weight.data = q_channels
    return layer

def qconv_pertensor(layer, nbits=8):
    tensor = layer.weight.data
    qmin = 0.
    qmax = 2.**nbits - 1.
    
    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    q_tensor = (tensor / scale + 0.5).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale + min_val
    layer.weight.data = q_tensor
    return layer


def qlin_pertensor(layer, nbits=8):
    tensor = layer.weight.data
    qmin = 0.
    qmax = 2.**nbits - 1.
    
    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin)
    q_tensor = (tensor / scale + 0.5).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale + min_val
    layer.weight.data = q_tensor
    return layer

def quantize_model(model): 
    qlayers = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            layer = convquant_perchannel(layer)
        if isinstance(layer, nn.Linear):
            layer = linquant_pertensor(layer)
        qlayers.append(layer)
    return nn.Sequential(*qlayers)



layer = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)

qlayer = qconv_perchannel(layer)
print(qlayer.weight.data.shape)