import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("results.csv")
data = data[data['precision'] == 'fp32_comp']

data[data["backbone"]]


"""

# Filter the data by precision = fp32_comp
filtered_data_fp32 = data[data['precision'] == 'fp32_comp']

# Calculate the required value for each combination of backbone and aggregation
filtered_data_fp32['value'] = filtered_data_fp32['pitts30k_r@1'] / filtered_data_fp32['model_size']

# Prepare data for plotting
plot_data_fp32_model_size = filtered_data_fp32.groupby(['backbone', 'aggregation'])['value'].mean().reset_index()

# Create a new column combining backbone and aggregation for the x-axis labels
plot_data_fp32_model_size['backbone_aggregation'] = plot_data_fp32_model_size['backbone'] + ' + ' + plot_data_fp32_model_size['aggregation']

# Plot the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x='backbone_aggregation', y='value', data=plot_data_fp32_model_size)
plt.title('Value of pitts30k_r@1 / model_size by Backbone and Aggregation (fp32_comp)')
plt.ylabel('Value (pitts30k_r@1 / model_size)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""