import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data = pd.read_csv("results.csv")
#data = data[data['backbone'] in ['efficientnet_b0', 'resnet50_conv4']
#data = data[data["aggregation"] == "gem"]


updated_retrieval_metrics = ['st_lucia_r@1', 'pitts30k_r@1', 'nordland_r@1']
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(15, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(data=data, x='backbone', y=metric, hue='precision', ax=ax[i], palette="Set2")
    ax[i].set_title(f'Distribution of {metric} by Backbone and Precision')
    ax[i].set_ylabel(f'{metric} Score')
    ax[i].set_xlabel('Backbone')
    ax[i].legend(title='Precision', loc='upper right')

plt.tight_layout()
plt.show()




# Plotting the effect of backbone on encoding latency and memory size
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Encoding Latency
sns.boxplot(data=data, x='backbone', y='mean_encoding_time', hue='precision', ax=ax[0], palette="Set2")
ax[0].set_title('Distribution of Mean Encoding Time by Backbone and Precision')
ax[0].set_ylabel('Mean Encoding Time (seconds)')
ax[0].set_xlabel('Backbone')
ax[0].legend(title='Precision', loc='upper right')

# Memory Size
sns.boxplot(data=data, x='backbone', y='model_size', hue='precision', ax=ax[1], palette="Set2")
ax[1].set_title('Distribution of Model Size by Backbone and Precision')
ax[1].set_ylabel('Model Size (bytes)')
ax[1].set_xlabel('Backbone')
ax[1].legend(title='Precision', loc='upper right')

plt.tight_layout()
plt.show()


plt.show()


# Plotting retrieval performance for updated metrics across aggregation methods
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(15, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(data=data, x='aggregation', y=metric, hue='precision', ax=ax[i], palette="Set3")
    ax[i].set_title(f'Distribution of {metric} by Aggregation and Precision')
    ax[i].set_ylabel(f'{metric} Score')
    ax[i].set_xlabel('Aggregation Method')
    ax[i].legend(title='Precision', loc='upper right')

plt.tight_layout()
plt.show()


# Scatter plot of mean_encoding_time vs. model_size colored by backbone
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='mean_encoding_time', y='model_size', hue='backbone', palette="deep", s=100, edgecolor="w", alpha=0.7)
plt.title('Mean Encoding Time vs. Model Size by Backbone')
plt.xlabel('Mean Encoding Time (seconds)')
plt.ylabel('Model Size (bytes)')
plt.legend(title='Backbone', loc='upper right')
plt.show()

# Box plot of model size across aggregation methods
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, x='aggregation', y='model_size', hue='precision', palette="muted")
plt.title('Distribution of Model Size by Aggregation and Precision')
plt.xlabel('Aggregation Method')
plt.ylabel('Model Size (bytes)')
plt.legend(title='Precision', loc='upper right')
plt.show()



# Box plot of mean encoding time across aggregation methods
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, x='aggregation', y='mean_encoding_time', hue='precision', palette="pastel")
plt.title('Distribution of Mean Encoding Time by Aggregation and Precision')
plt.xlabel('Aggregation Method')
plt.ylabel('Mean Encoding Time (seconds)')
plt.legend(title='Precision', loc='upper right')
plt.show()

heatmap_data = data.pivot_table(values='pitts30k_r@1', index='aggregation', columns='backbone', aggfunc='mean')
# Plot the heatmap with the 'coolwarm' color palette
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", cbar_kws={'label': 'Average pitts30k_r@1 Score'})
plt.title('Average pitts30k_r@1 Score by Backbone and Aggregation')
plt.xlabel('Backbone')
plt.ylabel('Aggregation Method')
plt.show()

# Pivot the data to get the mean scores for each metric based on backbone and aggregation
pitts30k_data = data.pivot_table(values='pitts30k_r@1', index='aggregation', columns='backbone', aggfunc='mean')
nordland_data = data.pivot_table(values='nordland_r@1', index='aggregation', columns='backbone', aggfunc='mean')
st_lucia_data = data.pivot_table(values='st_lucia_r@1', index='aggregation', columns='backbone', aggfunc='mean')

# Combine the heatmaps into a single figure
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Plot the heatmaps
sns.heatmap(pitts30k_data, annot=True, cmap="coolwarm", cbar_kws={'label': 'Avg pitts30k_r@1 Score'}, ax=ax[0])
sns.heatmap(nordland_data, annot=True, cmap="coolwarm", cbar_kws={'label': 'Avg nordland_r@1 Score'}, ax=ax[1])
sns.heatmap(st_lucia_data, annot=True, cmap="coolwarm", cbar_kws={'label': 'Avg st_lucia_r@1 Score'}, ax=ax[2])

# Set titles
ax[0].set_title('pitts30k_r@1 Score by Backbone and Aggregation')
ax[1].set_title('nordland_r@1 Score by Backbone and Aggregation')
ax[2].set_title('st_lucia_r@1 Score by Backbone and Aggregation')

plt.tight_layout()
plt.show()

