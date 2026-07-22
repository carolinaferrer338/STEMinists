import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns


# please commit
max_steps = 199000
step_size = 1000
ts = [x * step_size for x in range(1, int(max_steps / step_size) + 1)]

def generate_heat(state):
    comp_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ce_alpha_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    cty_alpha_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    metric_cols = ["PB", "CountySplits", "PP"]
    metric_dict = {"PB": {'color':"Blues", 'vmax':-0.2, 'vmin':0}, "CountySplits": {'color':"Greens", 'vmax':36, 'vmin':6}, "PP": {'color':"Purples", 'vmax':7.2, 'vmin':3.5}}

    for metric_col in metric_cols:
        rows = []

        for comp_alpha in comp_alpha_range:
            for ce_alpha in ce_alpha_range:
                for cty_alpha in cty_alpha_range:
                    for num in [1, 2]:

                        total = 0
                        count = 0

                        try:
                            df = pd.read_csv(
                                f'/Users/carolinaferrer/Downloads/tn_fixed/{state}_{comp_alpha}-{ce_alpha}-{cty_alpha}_200000_{num}/ensemble_{num}chain_outputs_1000.csv',
                                usecols=[metric_col]
                            )
                        except FileNotFoundError:
                            print(f"Missing file for {comp_alpha=}, {ce_alpha=}, {cty_alpha=}, {num=}")
                            continue

                        total += df[metric_col].sum()
                        count += len(df)

                        if count == 0:
                            continue

                        rows.append({
                            "comp_alpha": comp_alpha,
                            "ce_alpha": ce_alpha,
                            "cty_alpha": cty_alpha,
                            "avg_dw": total / count,
                            "ensemble": num
                        })

        summary = pd.DataFrame(rows)

        # Fixed County Splits Alpha (6 heatmaps)
        for cty_alpha in cty_alpha_range:

            plot_df = (
                summary[summary["cty_alpha"] == cty_alpha]
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
                vmin=int(metric_dict[metric_col]['vmin']),
                vmax=int(metric_dict[metric_col]['vmax']),
                annot=True,
                fmt=".3f",
                cmap=metric_dict[metric_col]['color'],
                linewidths=0.5
            )

            ax.invert_yaxis()

            plt.xlabel("Cut Edges Alpha")
            plt.ylabel("Competitiveness Alpha")
            plt.title(
                f"Average {metric_col} in {state.upper()} with County Surcharge = {cty_alpha}"
            )

            plt.savefig(
                f'/Users/carolinaferrer/STEMinists/Output/plots/{state.upper()}/{state}_heatmap_{metric_col}_CTY_{cty_alpha}.png',
                dpi=300
            )
            plt.close()

        # Fixed Cut Edges Alpha (5 heatmaps)
        for ce_alpha in ce_alpha_range:

            plot_df = (
                summary[summary["ce_alpha"] == ce_alpha]
                .groupby(["comp_alpha", "cty_alpha"])["avg_dw"]
                .mean()
                .reset_index()
            )

            heatmap_data = plot_df.pivot(
                index="comp_alpha",
                columns="cty_alpha",
                values="avg_dw"
            )

            plt.figure(figsize=(8, 6))

            ax = sns.heatmap(
                heatmap_data,
                vmin=metric_dict[metric_col]['vmin'],
                vmax=metric_dict[metric_col]['vmax'],
                annot=True,
                fmt=".3f",
                cmap=metric_dict[metric_col]['color'],
                linewidths=0.5
            )

            ax.invert_yaxis()

            plt.xlabel("County Splits Surcharge")
            plt.ylabel("Competitiveness Alpha")
            plt.title(
                f"Average {metric_col} in {state.upper()} with Cut Edges Alpha = {ce_alpha}"
            )

            plt.savefig(
                f'/Users/carolinaferrer/STEMinists/Output/plots/{state.upper()}/{state}_heatmap_{metric_col}_CE_{ce_alpha}.png',
                dpi=300
            )
            plt.close()

        # Fixed Competitiveness Alpha (6 heatmaps)
        for comp_alpha in comp_alpha_range:

            plot_df = (
                summary[summary["comp_alpha"] == comp_alpha]
                .groupby(["cty_alpha", "ce_alpha"])["avg_dw"]
                .mean()
                .reset_index()
            )

            heatmap_data = plot_df.pivot(
                index="cty_alpha",
                columns="ce_alpha",
                values="avg_dw"
            )

            plt.figure(figsize=(8, 6))

            ax = sns.heatmap(
                heatmap_data,
                vmin=metric_dict[metric_col]['vmin'],
                vmax=metric_dict[metric_col]['vmax'],
                annot=True,
                fmt=".3f",
                cmap=metric_dict[metric_col]['color'],
                linewidths=0.5
            )

            ax.invert_yaxis()

            plt.xlabel("Cut Edges Alpha")
            plt.ylabel("County Splits Surcharge")
            plt.title(
                f"Average {metric_col} in {state.upper()} with Competitiveness Alpha = {comp_alpha}"
            )

            plt.savefig(
                f'/Users/carolinaferrer/STEMinists/Output/plots/{state.upper()}/{state}_heatmap_{metric_col}_COMP_{comp_alpha}.png',
                dpi=300
            )
            plt.close()

generate_heat('tn')
