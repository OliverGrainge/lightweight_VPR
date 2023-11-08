from QuantizationSim import Quantizer 
from torchvision.models import resnet18
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from model import network
import util
import datasets_ws
import parser
import test



args = parser.parse_arguments()

args.datasets_folder = "/Users/olivergrainge/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.dataset_name = "pitts30k"
args.device = "cpu"
args.resume = "/Users/olivergrainge/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_mac_1024/best_model.pth"
args.backbone = "mobilenetv2conv4"
args.fc_output_dim = 1024
args.aggregation = "mac"
args.num_workers = 0

model = network.GeoLocalizationNet(args).cpu().eval()
model = util.resume_model(args, model)
qmodel = network.GeoLocalizationNet(args).cpu().eval()
qmodel = util.resume_model(args, qmodel)

class CalibrationDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
    
    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx]

cal_ds = CalibrationDataset(torch.randn(1, 3, 224, 224))
cal_dl = DataLoader(cal_ds, batch_size=1)

quantizer = Quantizer(qmodel,
    layer_precision="fp16",
    activation_precision="fp16",
    activation_granularity="tensor",
    layer_granularity="channel",
    calibration_type="minmax", 
    calibration_loader=cal_dl,
    activation_layers = (nn.ReLU, nn.ReLU6, nn.BatchNorm2d))

x = torch.randn(1, 3, 480, 640)

qmodel = quantizer.fit()
print("Layer Quantized ", quantizer.layers_quantized)
print("Layer Count", quantizer.num_layers())
print("Activation Count", quantizer.num_activations())
print("Activations Quantized", quantizer.activations_quantized)
print("Number Activation Collected:", len(quantizer.activation_data))

desc = model(x).flatten()
qdesc = qmodel(x).flatten()


test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
recalls, retrieval_time, feature_bytes, recalls_str = test.test(
    args, test_ds, model, args.test_method, None
)