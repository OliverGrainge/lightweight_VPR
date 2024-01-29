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

DATASET_NAMES = ["nordland", "st_lucia"]
WEIGHT_PRECISION = ["int4", "int8", "fp16", "fp32"]
CALIBRATION = ["percentile", "minmax"]
DESCRIPTOR_SIZE = {'512', '1024', '2048', '4096'}
WEIGHT_GRANULARITY = {'tensor', 'channel'}


args = parser.parse_arguments()


def run_experiment(exp_config):
    args.aggregation = exp_config["aggregation"]
    args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
    args.backbone = "mobilenetv2conv4"
    args.dataset_name = exp_config["dataset_name"]
    args.fc_output_dim = int(exp_config["fc_output_dim"])
    args.infer_batch_size=3
    args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{exp_config['aggregation']}_{exp_config['fc_output_dim']}/best_model.pth"
    qmodel = network.GeoLocalizationNet(args).cpu().eval()
    qmodel = util.resume_model(args, qmodel)

    cal_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    cal_dl = DataLoader(cal_ds, batch_size=1)

    if exp_config["layer_precision"].startswith('int'):
        activation_precision = "int32"
    else: 
        activation_precision = exp_config["layer_precision"]

    quantizer = Quantizer(qmodel.cpu(),
        layer_precision=exp_config["layer_precision"],
        activation_precision=activation_precision,
        activation_granularity="tensor",
        layer_granularity=exp_config["layer_granularity"],
        calibration_type=exp_config["calibration_type"], 
        cal_samples=100,
        calibration_loader=cal_dl,
        activation_layers = (nn.ReLU, nn.ReLU6), 
        device="cpu")

    qmodel = quantizer.fit()
    
    print("Layer Quantized ", quantizer.layers_quantized)
    print("Layer Count", quantizer.num_layers())
    print("Activation Count", quantizer.num_activations())
    print("Activations Quantized", quantizer.activations_quantized)
    print("Number Activation Collected:", len(quantizer.activation_data))
    del quantizer
    del cal_ds
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    recalls, retrieval_time, feature_bytes, recalls_str = test.test(
        args, test_ds, qmodel.cuda(), args.test_method, None
    )
    del test_ds
    del qmodel
    return recalls[0]

import pandas as pd



for dataset in DATASET_NAMES:
    for layer_precision in WEIGHT_PRECISION:
        for fc_output_dim in DESCRIPTOR_SIZE:
            for granularity in WEIGHT_GRANULARITY:
                for calibration_type in CALIBRATION:
                    try:
                        df = pd.read_csv("quantization_results.csv")
                    except: 
                        df = None
                    print("==============================================================")
                    print(df)

                    exp_config = {"dataset_name":dataset, 
                                "layer_precision":layer_precision,
                                "fc_output_dim":fc_output_dim,
                                "layer_granularity":granularity,
                                "calibration_type": calibration_type,
                                "aggregation": "mac"}
                    
                    rat1 = run_experiment(exp_config)

                    exp_config["R@1"] = rat1
                    result = exp_config

                    print(result)

                    try:
                        new_df = pd.DataFrame([result], index=None)
                        print(new_df)
                        df = pd.concat([df, new_df])
                    except:
                        df = pd.DataFrame([result])

                    df.to_csv("quantization_results.csv", columns=["dataset_name", "layer_precision", "fc_output_dim", "layer_granularity", "calibration_type", "aggregation", "R@1"])

