import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")


def get_recallvsprecision(df):
    precisions = [1.0, 2.0, 3.0]
    prec_label = ["fp32_comp", "fp16_comp", "int8_comp"]
    recalls = [
        df[df["precision"] == prec]["pitts30k_r@1"].to_numpy()[0] for prec in prec_label
    ]
    return precisions, recalls


efficient_df = df[df["id"].str.startswith("efficientnet")]

resnet18_df = df[df["id"].str.startswith("resnet18")]

resnet50_df = df[df["id"].str.startswith("resnet50")]

mobilenet_df = df[df["id"].str.startswith("mobilenet")]


fig, ax = plt.subplots(1)

precision, recalls = get_recallvsprecision(efficient_df)
print(precision, recalls)
print(precision, recalls)
ax.plot(precision, recalls, label="efficientnet_b0")

precision, recalls = get_recallvsprecision(resnet18_df)
ax.plot(precision, recalls, label="resnet18")

precision, recalls = get_recallvsprecision(resnet50_df)
ax.plot(precision, recalls, label="resnet50")

precision, recalls = get_recallvsprecision(mobilenet_df)
ax.plot(precision, recalls, label="mobilenetv2")

months = ["fp32", "fp16", "int8"]
values = [1.0, 2.0, 3.0]
ax.set_xticks(ticks=range(1, 4), labels=months)
plt.legend()
ax.set_ylabel("Recall@1")
ax.set_xlabel("Precision")
ax.set_title("Pitts30k")
plt.show()
