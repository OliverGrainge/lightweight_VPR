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

args = parser.argparse()

args.aggregation = exp_config["aggregation"]
args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.backbone = "mobilenetv2conv4"
args.dataset_name = exp_config["dataset_name"]
args.fc_output_dim = int(exp_config["fc_output_dim"])
args.infer_batch_size=3
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{exp_config['aggregation']}_{exp_config['fc_output_dim']}/best_model.pth"
qmodel = network.GeoLocalizationNet(args).cpu().eval()
qmodel = util.resume_model(args, qmodel)
model = network.GeoLocalizationNet(args).cpu().eval()
model = util.resume_model(args, model)

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")


def init_pop(popsize: int=100) -> list:
    cal_dl = DataLoader(test_ds, batch_size=4)

    quantizer = Quantizer(qmodel.cpu(),
        layer_precision="fp32",
        activation_precision="fp32",
        activation_granularity="tensor",
        layer_granularity="tensor",
        calibration_type="minmax", 
        cal_samples=100,
        calibration_loader=cal_dl,
        activation_layers = (nn.ReLU, nn.ReLU6), 
        device="cpu")
    
    num_layers = quantizer.num_layers()
    return num_layers
print(init_pop())



def mutation(individual: list) -> list:
    pass 



def cost(q_features: torch.tensor, target_features: torch.tensor) -> float:
    pass 



def selection(population: list) -> list: 
    # returns best parent 
    pass 