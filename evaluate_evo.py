import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import parser 
from model import network 
import util
from QuantizationSim import Quantizer 
import torch.nn as nn
import datasets_ws
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

average_bitwidth = 10




fitness = np.load(f"/home/oliver/Documents/github/lightweight_VPR/evoquant_data/mobilenetv2conv4_mac_1024_{str(average_bitwidth)}/all_fitness.npy")

# Define the window size for the SMA
window_size = 5

# Calculate the SMA
smoothed_y = np.convolve(fitness, np.ones(window_size)/window_size, mode='same')
smoothed_y = savgol_filter(fitness, window_length=11, polyorder=3)
x = np.arange(len(smoothed_y))
# Plot the original and smoothed data
plt.figure(figsize=(10, 5))
plt.plot(x, smoothed_y, label=f'Smoothed (SMA, {window_size}-point)', color='red', linewidth=2)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Smoothing with SMA')
plt.grid(True)




args = parser.parse_arguments()
args.aggregation = "mac"
args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.backbone = "mobilenetv2conv4"
args.dataset_name = "pitts30k"
args.fc_output_dim = 1024
args.infer_batch_size=3
args.device = 'cuda:0'
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{args.aggregation}_{str(args.fc_output_dim)}/best_model.pth"
model = network.GeoLocalizationNet(args).eval().to(args.device)
model = util.resume_model(args, model)


test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
test_dl = DataLoader(test_ds, 5)

quantizer = Quantizer(model,
        layer_precision="fp32",
        activation_precision="fp32",
        activation_granularity="tensor",
        layer_granularity="channel",
        calibration_type="minmax", 
        cal_samples=10,
        calibration_loader=test_dl,
        activation_layers = (nn.ReLU, nn.ReLU6), 
        device=args.device, 
        number_of_activations=32,
        number_of_layers=103)


quantizer.quantize_layers()



ata = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value': [10, 15, 10, 20, 5, 25],
    'Group': ['G1', 'G2', 'G1', 'G2', 'G1', 'G2']
})

layer_types = list(quantizer.all_layer_types)

layer_types = [item for item in layer_types if item != "BatchNorm2d"]

plt.figure()
# Create a bar plot with Seaborn





plt.title("MobileNetV2 Mixed-Precision Configuration")
prec = np.load(f"/home/oliver/Documents/github/lightweight_VPR/evoquant_data/mobilenetv2_mac_1024/best_configuration.npy")
values = {"int4":4, "int8":8, "fp16":16}
data = [values[gene] for gene in prec]
df = pd.DataFrame.from_dict({"Bit-width":data, "layer index":np.arange(len(data)), "Group": layer_types})
ax = sns.barplot(x='layer index', y='Bit-width', hue='Group', data=df, palette='deep')
current_axis = plt.gca()

# Set xticks such that only every 5th label is shown
current_axis.set_xticks(current_axis.get_xticks()[::5])
plt.xlabel("Layer Index")
plt.ylabel("Layer Bit-Width")
handles, labels = ax.get_legend_handles_labels()
new_labels = ['Conv2d 3x3', 'Conv2d 1x1', 'Linear']
ax.legend(handles, new_labels)
#plt.legend()

import pandas as pd
df = pd.read_csv("evo_quant_line_plot.csv")
df2 = pd.read_csv("evo_quant_line_plot_stlucia.csv")
df3 = pd.read_csv("evo_quant_line_plot_nordland.csv")



plt.figure()
plt.plot(df["average_bitwidth"], df["recall@1"], marker='x', label="Pitts30k")
plt.plot(df2["average_bitwidth"], df2["recall@1"], marker='x', label='St Lucia')
plt.plot(df3["average_bitwidth"], df3["recall@1"], marker='x', label="Nordland")
plt.legend(title="Dataset")
plt.xlabel("Average Bit-width")
plt.ylabel("Recall@1")
plt.title("Recall vs Mixed-Precsion Average Bit-Width")
plt.gca().invert_xaxis()
plt.show()

