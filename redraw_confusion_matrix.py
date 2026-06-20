import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "redraw.csv"))

output_dir = os.path.join(os.path.dirname(__file__), "redraw")
os.makedirs(output_dir, exist_ok=True)

for _, row in df.iterrows():
    filename = row["filename"]
    values = row.iloc[1:26].values.astype(int)
    matrix = values.reshape(5, 5)

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=False,
        yticklabels=False,
        annot_kws={"size": 50},
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=50)
    ax.set_ylabel("Actual", fontsize=50)
    ax.tick_params(axis="both", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

print("Done")

