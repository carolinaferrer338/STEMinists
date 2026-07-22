import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns

max_steps = 199000
step_size = 1000
ts = [x * step_size for x in range(1, int(max_steps / step_size) + 1)]

def generate_heat(state, cty_alpha):
    rows = []
    for comp_alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for ce_alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for num in [1, 2]:

                total = 0
                count = 0

                for t in ts:
                    try:
                        df = pd.read_csv(
                            f'/Users/carolinaferrer/Downloads/tn_fixed/{state}_{comp_alpha}-{ce_alpha}-{cty_alpha}_200000_{num}/ensemble_{num}chain_outputs_1000.csv',
                            usecols=["PB"]
                        )
                    except FileNotFoundError:
                        print(f"Missing file for {comp_alpha=}, {ce_alpha=}, {cty_alpha=}, {num=}")
                        continue

                    total += df["PB"].sum()
                    count += len(df)

                if count == 0:
                    continue

                rows.append({
                    "comp_alpha": comp_alpha,
                    "ce_alpha": ce_alpha,
                    "avg_dw": total / count,
                    "ensemble": num
                })

    summary = pd.DataFrame(rows)
    plot_df = (
        summary
        .groupby(["comp_alpha", "ce_alpha"])["avg_dw"]
        .mean()
        .reset_index()
    )
    heatmap_data = plot_df.pivot(
        index="comp_alpha",
        columns="ce_alpha",
        values="avg_dw"
    )
    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        heatmap_data,
        vmin= -0.2, 
        vmax= 0,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=0.5
    )
    ax.invert_yaxis()

    plt.xlabel("Cut Edges Alpha")
    plt.ylabel("Competitiveness Alpha")
    plt.title(f"Average Partisan Bias in {state.upper()} with County Surcharge = {cty_alpha}")
    plt.savefig(f'/Users/carolinaferrer/STEMinists/Output/plots/{state.upper()}/{state}_heatmap_pb_{cty_alpha}.png', dpi=300)

for cty in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    generate_heat('tn', cty)