import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("results.csv")

# data = data[data['backbone'] in ['efficientnet_b0', 'resnet50_conv4']
# data = data[data["aggregation"] == "gem"]

updated_retrieval_metrics = ["st_lucia_r@1", "pitts30k_r@1", "nordland_r@1"]
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(15, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(
        data=data, x="backbone", y=metric, hue="precision", ax=ax[i], palette="Set2"
    )
    ax[i].set_title(
        f"Distribution of {metric} by Backbone and Precision", fontsize="18"
    )
    ax[i].set_ylabel(f"{metric} Score", fontsize="15")
    ax[i].set_xlabel("Backbone", fontsize="15")
    ax[i].legend(title="Precision", loc="upper right")


# Plotting the effect of backbone on encoding latency and memory size
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Encoding Latency
sns.boxplot(
    data=data,
    x="backbone",
    y="mean_encoding_time",
    hue="precision",
    ax=ax[0],
    palette="Set2",
)
ax[0].set_title("Feature Encoding Time by Backbone and Precision", fontsize="19")
ax[0].set_ylabel("Encoding Time (seconds)", fontsize="16")
ax[0].set_xlabel("Backbone")
ax[0].legend(title="Precision", loc="upper right")
# Memory Size
sns.boxplot(
    data=data, x="backbone", y="model_size", hue="precision", ax=ax[1], palette="Set2"
)
ax[1].set_title("Distribution of Model Size by Backbone and Precision")
ax[1].set_ylabel("Model Size (bytes)")
ax[1].set_xlabel("Backbone")
ax[1].legend(title="Precision", loc="upper right")

plt.tight_layout()


# Plotting retrieval performance for updated metrics across aggregation methods
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(15, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(
        data=data, x="aggregation", y=metric, hue="precision", ax=ax[i], palette="Set3"
    )
    ax[i].set_title(f"Distribution of {metric} by Aggregation and Precision")
    ax[i].set_ylabel(f"{metric} Score")
    ax[i].set_xlabel("Aggregation Method")
    ax[i].legend(title="Precision", loc="upper right")


# Scatter plot of mean_encoding_time vs. model_size colored by backbone
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=data,
    x="mean_encoding_time",
    y="model_size",
    hue="backbone",
    palette="deep",
    s=100,
    edgecolor="w",
    alpha=0.7,
)
plt.title("Mean Encoding Time vs. Model Size by Backbone")
plt.xlabel("Mean Encoding Time (seconds)")
plt.ylabel("Model Size (bytes)")
plt.legend(title="Backbone", loc="upper right")

# Box plot of model size across aggregation methods
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=data, x="aggregation", y="model_size", hue="precision", palette="muted"
)
plt.title("Distribution of Model Size by Aggregation and Precision")
plt.xlabel("Aggregation Method")
plt.ylabel("Model Size (bytes)")
plt.legend(title="Precision", loc="upper right")


# Box plot of mean encoding time across aggregation methods
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=data,
    x="aggregation",
    y="mean_encoding_time",
    hue="precision",
    palette="pastel",
)
plt.title("Feature Encoding Time by Aggregation and Precision", fontsize="16")
plt.xlabel("Aggregation Method", fontsize="14")
plt.ylabel("Encoding Time (ms)", fontsize="14")
plt.legend(title="Precision", loc="upper right")

heatmap_data = data.pivot_table(
    values="pitts30k_r@1", index="aggregation", columns="backbone", aggfunc="mean"
)
# Plot the heatmap with the 'coolwarm' color palette
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Average pitts30k_r@1 Score"},
)
plt.title("Average Pitts30k R@1 Score by Backbone and Aggregation")
plt.xlabel("Backbone")
plt.ylabel("Aggregation Method")

# plt.show()


data = data[data["precision"] == "int8_comp"]
print(data.keys())
# Pivot the data to get the mean scores for each metric based on backbone and aggregation
pitts30k_data = data.pivot_table(
    values="pitts30k_r@1", index="aggregation", columns="backbone", aggfunc="max"
)
nordland_data = data.pivot_table(
    values="nordland_r@1", index="aggregation", columns="backbone", aggfunc="max"
)
st_lucia_data = data.pivot_table(
    values="st_lucia_r@1", index="aggregation", columns="backbone", aggfunc="max"
)


# Combine the heatmaps into a single figure
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Plot the heatmaps
sns.heatmap(
    pitts30k_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Avg pitts30k_r@1 Score"},
    ax=ax[0],
)
sns.heatmap(
    nordland_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Avg nordland_r@1 Score"},
    ax=ax[1],
)
sns.heatmap(
    st_lucia_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Avg st_lucia_r@1 Score"},
    ax=ax[2],
)

# Set titles
ax[0].set_title("Pitts30k R@1 Score by Backbone and Aggregation")
ax[1].set_title("Nordland R@1 Score by Backbone and Aggregation")
ax[2].set_title("St Lucia R@1 Score by Backbone and Aggregation")

plt.tight_layout()
# plt.show()


filtered_data = data[data["backbone"] == "mobilenetv2conv4"]
# Filter the data further to only include rows with the "GEM" aggregation type
gem_filtered_data = filtered_data[filtered_data["aggregation"] == "gem"]
# print(gem_filtered_data.keys())
# gem_filtered_data["backbone", "aggregation", "descriptor_size", "precision"]
print(gem_filtered_data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(
    gem_filtered_data["descriptor_size"],
    gem_filtered_data["pitts30k_r@1"],
    color="blue",
    alpha=0.7,
)
plt.title("Influence of Descriptor Size on Pitts30k R@1 Performance (GEM Aggregation)")
plt.xlabel("Descriptor Size")
plt.ylabel("Pitts30k R@1 (%)")
plt.grid(True)
plt.tight_layout()


backbone = "mobilenetv2conv4"
aggregation = "mac"
df = pd.read_csv("results.csv")

dataset = "pitts30k_r@1"
df = df[df["backbone"] == backbone]
df = df[df["aggregation"] == aggregation]

df = df[["precision", "fc_output_dim", "pitts30k_r@1", "nordland_r@1", "st_lucia_r@1"]]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
precision_mapping = {"fp32_comp": 1, "fp16_comp": 2, "int8_comp": 3}
df["precision_numeric"] = df["precision"].map(precision_mapping)
# Scatter plot
sc = ax.scatter(
    df["precision_numeric"],
    df["fc_output_dim"],
    df[dataset],
    c=df[dataset],
    cmap="viridis",
    s=60,
)

# Surface plot
ax.plot_trisurf(
    df["precision_numeric"], df["fc_output_dim"], df[dataset], cmap="viridis", alpha=0.5
)


ax.set_xlabel("Precision")
ax.set_ylabel("fc_output_dim")
ax.set_zlabel(dataset)
ax.set_xticks(list(precision_mapping.values()))
ax.set_xticklabels(list(precision_mapping.keys()))
ax.view_init(elev=80, azim=0)

# Colorbar
cbar = fig.colorbar(sc)
cbar.ax.set_ylabel("pitts30k_r@1 values")

plt.title("3D Surface Plot of precision, fc_output_dim against pitts30k_r@1")
plt.show()
