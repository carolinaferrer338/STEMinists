#all the imports needed for this file
from typing import List, Dict
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations
import math
import warnings
import glob
from matplotlib.colors import LinearSegmentedColormap

DISTRICTS_BY_STATE = {
    "CO": {"congress": 8, "upper": 35, "lower": 65},
    "MA": {"congress": 9, "upper": 40, "lower": 160},
    "NV": {"congress": 4, "upper": 21, "lower": 42},
    "PA": {"congress": 17, "upper": 50, "lower": 203},
    "TN": {"congress": 9, "upper": 33, "lower": 99},
    "UT": {"congress": 4, "upper": 29, "lower": 75},
}

state_list = ["PA", "MA", "UT", "TN", "CO"]
 
state_chamber_list = [("CO", "congress")]

MAX_STEPS = 199000
STEP_SIZE = 1000

debug: bool = False  # Set to True to run the test code at the end of this file


#names of ensembles and also formatting for the ensembles inside of the plots/ tables

my_ensemble_list = [
    "co_1-1-1_200000_1",
    "co_0.9-1-1_200000_1",
    "co_0.8-1-1_200000_1",
    "co_0.7-1-1_200000_1",
    "co_0.6-1-1_200000_1"
]

_ensemble_mapping: Dict[str, str] = {
    "PA_comp_06_01": "PA_comp_06_01",
    "PA_comp_06_02": "PA_comp_06_02",
    "PA_comp_07_01": "PA_comp_07_01",
    "PA_comp_07_02": "PA_comp_07_02",
    "PA_comp_08_01": "PA_comp_08_01",
    "PA_comp_08_02": "PA_comp_08_02",
    "PA_comp_09_01": "PA_comp_09_01",
    "PA_comp_09_02": "PA_comp_09_02",
    "PA_comp1_01": "PA_comp1_01",
    "PA_comp1_02": "PA_comp1_02",
}

RECOM_ENSEMBLES = {
    "co_1-1-1_200000_1": "Output/co_1-1-1_200000_1/ensemble_1chain_outputs_* (1).csv",
    "co_0.9-1-1_200000_1": "Output/co_0.9-1-1_200000_1/ensemble_1chain_outputs_* (2).csv",
    "co_0.8-1-1_200000_1": "Output/co_0.8-1-1_200000_1/ensemble_1chain_outputs_* (3).csv",
    "co_0.7-1-1_200000_1": "Output/co_0.7-1-1_200000_1/ensemble_1chain_outputs_* (4).csv",
    "co_0.6-1-1_200000_1": "Output/co_0.6-1-1_200000_1/ensemble_1chain_outputs_* (5).csv"
}

RECOM_DEM_ENSEMBLES = {
    "co_1-1-1_200000_1": "Output/co_1-1-1_200000_1/ensemble_1_DemPercs_* (1).csv",
    "co_0.9-1-1_200000_1": "Output/co_0.9-1-1_200000_1/ensemble_1_DemPercs_* (2).csv",
    "co_0.8-1-1_200000_1": "Output/co_0.8-1-1_200000_1/ensemble_1_DemPercs_* (3).csv",
    "co_0.7-1-1_200000_1": "Output/co_0.7-1-1_200000_1/ensemble_1_DemPercs_* (4).csv",
    "co_0.6-1-1_200000_1": "Output/co_0.6-1-1_200000_1/ensemble_1_DemPercs_* (5).csv"
}

RECOM_MMD_ENSEMBLES = {
    "co_1-1-1_200000_1": "Output/co_1-1-1_200000_1/ensemble_1mmd_outputs_* (1).csv",
    "co_0.9-1-1_200000_1": "Output/co_0.9-1-1_200000_1/ensemble_1mmd_outputs_* (2).csv",
    "co_0.8-1-1_200000_1": "Output/co_0.8-1-1_200000_1/ensemble_1mmd_outputs_* (3).csv",
    "co_0.7-1-1_200000_1": "Output/co_0.7-1-1_200000_1/ensemble_1mmd_outputs_* (4).csv",
    "co_0.6-1-1_200000_1": "Output/co_0.6-1-1_200000_1/ensemble_1mmd_outputs_* (5).csv"
}

# convert to names compatable with Matplotlib
ensemble_name_dict_for_plots = {
    "co_1-1-1_200000_1": "$co_{1.0}$",
    "co_0.9-1-1_200000_1": "$co_{0.9}$",
    "co_0.8-1-1_200000_1": "$co_{0.8}$",
    "co_0.7-1-1_200000_1": "$co_{0.7}$",
    "co_0.6-1-1_200000_1": "$co_{0.6}$",
}





#scores that are used for comparison
scores_output_names = {
    "county splits": "CountySplits",
    "mean-median": "MM",
    "efficiency gap": "EG",
    "seat bias": "PB",
    "competitive districts": "Comp45-55",
    "Polsby-Popper": "PP",
    "Dem seats": "DWins",
}

mmd_score_list = {
    "Opportunity districts": "Opportunity districts",
    "Coalition districts": "Coalition districts",
    "Proportional Opportunity": "Proportional Opportunity",
    "Proportional Coalition": "Proportional Coalition"
}




#getting seats from the DISTRICTS_BY_STATE (not super usefull since I have it written at the top but dont want to mess format)
def _extract_num_seats(districts_by_state):
    """Extract num_seats_dict from DISTRICTS_BY_STATE."""

    num_seats_dict = {}

    for state, state_data in districts_by_state.items():
        for chamber, seats in state_data.items():
            if seats is not None:  # Skip None values
                num_seats_dict[(state, chamber)] = seats

    return num_seats_dict

#load the recom scores from the csv for the ensembles. Each score is put into its own seperate arrays 
def load_recom_scores(ensemble, score_name):
    folder = RECOM_ENSEMBLES[ensemble]

    files = sorted(glob.glob(folder))

    if len(files) == 0:
        raise ValueError(f"No chain_outputs files found in {folder}")


    if score_name not in scores_output_names:
        raise ValueError(f"Score {score_name} not supported for recom ensembles.")

    col = scores_output_names[score_name]

    # Combine all CSVs into one long array
    arrays = []
    for f in files:
        df = pd.read_csv(f)
        arrays.append(df[col].values)

    return np.concatenate(arrays)

def load_mmd_scores(ensemble, score):
    folder = RECOM_MMD_ENSEMBLES[ensemble]

    files =  sorted(glob.glob(folder))

    if len(files) == 0:
        raise ValueError(f"No mmd_outputs files found in {folder}")

    if score not in mmd_score_list:
        raise ValueError(f"Score {score} not supported for recom ensembles.")

    col = mmd_score_list[score]

    arrays = []
    for f in files:
        df = pd.read_csv(f)
        arrays.append(df[col].values)

    return np.concatenate(arrays)

def load_dem_percents(ensemble):
    folder = RECOM_DEM_ENSEMBLES[ensemble]

    files =  sorted(glob.glob(folder))

    if len(files) == 0:
        raise ValueError(f"No chain_outputs files found in {folder}")

    dfs = []
    for f in files:
        if os.path.getsize(f) == 0:
            print(f"WARNING: Empty file skipped:", f)
            continue
        df2 = pd.read_csv(f)
        dfs.append(df2)

    df = pd.concat(dfs, ignore_index = True)

    return df
    # return df.to_numpy()

def load_filter_rhat(compare_results, score, top_chains):
    return {
        key: value
        for key, value in compare_results["rhat"].items()
        if key[2] == score and (key[0] in top_chains or key[1] in top_chains)
    }


def make_rhat_matrix(filtered_rhat, top_chains, ensembles):
    matrix = pd.DataFrame(index=top_chains, columns=ensembles, dtype=float)

    for key, value in filtered_rhat.items():
        e1, e2, score = key

        if e1 in top_chains:
            matrix.loc[e1, e2] = value
        if e2 in top_chains:
            matrix.loc[e2, e1] = value

    # Fill diagonal for the two top chains
    for chain in top_chains:
        matrix.loc[chain, chain] = 1.0

    matrix = matrix.fillna(1.0)

    return matrix

#get the score that is needed using the load score functions 
def fetch_score_array(state, chamber, ensemble_type, score):

    folder = RECOM_ENSEMBLES[ensemble_type]

    # MMD scores
    if score in mmd_score_list:
        mmd_df = load_mmd_scores(ensemble_type)
        return mmd_df

    # District-level Democratic vote share
    if score == "DemPercs":
        dem_df = load_dem_percents(state, chamber, ensemble_type)
        return dem_df

    # Plan-level scores
    if score not in scores_output_names:
        raise ValueError(f"Score {score} not supported.")

    output_df = load_recom_scores(ensemble_type, score)

    return output_df





#statistic test for mapping different enesembles onto the same graphs 
def t_test(a0, a1): 
    # runs the t-test of the hypotheses that two arrays were drawn from distributions with the same means.
    result = stats.ttest_ind(a0, a1, equal_var=False)
    return (
        result.statistic,
        result.pvalue,
    )  # the statistic is positive if a0 has a larger mean than a1

def ks_test(a0, a1):  
    # runs the Kolmogorov-Smirnov test that the two arrays were drawn from the same distribution
    result = stats.ks_2samp(a0, a1)
    return (
        result.statistic,
        result.pvalue,
        result.statistic_sign,
    )  # the statistic_sign is positive if a1 has larger values than a0

def gelman_rubin_rhat(a1, a2):
    n = len(a1)
    assert len(a2) == n, "Both chains must have the same length"

    # Means and variances
    mu1, mu2 = np.mean(a1), np.mean(a2)
    s1_sq, s2_sq = np.var(a1, ddof=1), np.var(a2, ddof=1)

    W = (s1_sq + s2_sq) / 2
    B = n * ((mu1 - mu2) ** 2) / 2
    V_hat = ((n - 1) / n) * W + (1 / n) * B
    R_hat = np.sqrt(V_hat / W)

    return R_hat






#functions for plotting the different scores

#compares all ensambles scores on a single line plot using the kde stat
def kde_plot(state, chamber, ensemble_list, score, average_lines=True, filename=None, ax=None):
    """
    For the given state, chamber, and score, this plots one KDE for each ensemble in ensemble_list.
    """
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots()
        created_ax = True

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ensemble in enumerate(ensemble_list):
        color = prop_cycle[i % len(prop_cycle)]
        a = fetch_score_array(state, chamber, ensemble, score)
        sns.kdeplot(a, label=ensemble_name_dict_for_plots[ensemble], color=color, ax=ax)
        if average_lines:
            ax.axvline(np.mean(a), linestyle="--", color=color)

    ax.set_title(f"{state} {chamber}: {score}")
    ax.set_xlabel(score)
    ax.set_ylabel("Density")
    ax.legend(title="Ensemble")

    if filename is not None:
        plt.savefig(filename)

    if created_ax:
        plt.show()

#compares all ensembles scores in block plot format for the score given
def box_whisker_plot(state, chamber, ensemble_list, score, filename=None):  # box plot of any given list of ensembles
    """
    For the given state, chamber, and score, this plots one box plot for each ensembles in ensemble_list.
    """
    data = pd.DataFrame(columns=["ensemble", "score"])
    for ensemble in ensemble_list:
        a = fetch_score_array(state, chamber, ensemble, score)
        data_for_a = pd.DataFrame({"ensemble": [ensemble] * len(a), "score": a})
        data = pd.concat([data, data_for_a], ignore_index=True)
    ax = sns.boxplot(data=data, x="ensemble", y="score", hue="ensemble", fliersize=0)
    ax.set_title(f"{state} {chamber}: {score} boxplots by ensemble type")
    ax.set_xlabel("Ensemble")
    ax.set_ylabel(score)

    # Set LaTeX labels for each tick on the x-axis
    ax.set_xticklabels(
        [ensemble_name_dict_for_plots[ensemble] for ensemble in ensemble_list],
        rotation=45,
    )
    ax.grid(axis="y")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    plt.show()

#plots one box plot per score across all ensemble. plots arrange in a grid with a number of columns
def box_whisker_plots_grid(state, chamber, ensemble_list, score_list, filename=None, cols=2):
    """
    For the given state and chamber, this plots one box plot per score across all ensembles.
    Plots are arranged in a grid with a configurable number of columns.

    Parameters:
        state (str): The state name.
        chamber (str): The chamber name.
        ensemble_list (list of str): List of ensemble names.
        score_list (list of str): List of scores to plot.
        filename (str, optional): If provided, saves the plot to this file.
        cols (int): Number of columns in the subplot grid.
    """
    num_scores = len(score_list)
    rows = math.ceil(num_scores / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = (
        axes.flatten() if num_scores > 1 else [axes]
    )  # Make sure axes is always iterable

    for idx, score in enumerate(score_list):
        data = pd.DataFrame(columns=["ensemble", "score"])
        for ensemble in ensemble_list:
            a = fetch_score_array(state, chamber, ensemble, score)
            data_for_a = pd.DataFrame({"ensemble": [ensemble] * len(a), "score": a})
            data = pd.concat([data, data_for_a], ignore_index=True)

        ax = axes[idx]
        sns.boxplot(
            data=data, x="ensemble", y="score", hue="ensemble", fliersize=0, ax=ax
        )
        ax.set_title(f"{score}")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticklabels(
            [ensemble_name_dict_for_plots[ensemble] for ensemble in ensemble_list],
            rotation=45,
        )
        ax.grid(axis="y")

    # Hide any unused axes
    for j in range(len(score_list), len(axes)):
        fig.delaxes(axes[j])

    # Add a single centered title for the whole grid
    # plt.tight_layout()
    fig.suptitle(f"{state} {chamber}: Boxplots of scores by Ensemble Type", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

    if filename is not None:
        plt.savefig(filename)
    plt.show()

#compares 2 ensembles democratic/ repulbican percentages on a ordered plot
def ordered_seats_plot(state, chamber, ensemble_list, competitive_window=0.05, filename=None):
    """
    Creates an ordered seats plot comparing Democratic vote share distributions
    across districts for multiple ensembles.

    Parameters:
        state (str): State abbreviation, chamber (str): Chamber name. ensemble_list: List of ensemble names to compare.
        competitive_window: Margin around 0.5 for dashed reference lines. filename: If provided, saves the plot to this file.
    """

    # load dem vote share for each ensemble
    x_data = {}

    for ensemble in ensemble_list:
        files = sorted(glob.glob(RECOM_DEM_ENSEMBLES[ensemble]))

        dfs = [pd.read_csv(f, header=None) for f in files]
        combined_df = pd.concat(dfs, ignore_index=True)
        x_data[ensemble] = combined_df.to_numpy()

    # get number of districts
    num_districts = x_data[ensemble_list[0]].shape[1]

    # mkaing the figureeee
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["lightblue", "lightgreen", "lightcoral", "lightgray", "lightyellow"]

    # plotting boxplots for each ensemble
    for idx, ensemble in enumerate(ensemble_list):
        X = x_data[ensemble]
        offset = (idx - len(ensemble_list)/2) * 0.3  # spacing between ensembles
        for i in range(num_districts):
            ax.boxplot(
                X[:, i],
                positions=[i + 1 + offset],
                widths=0.25,
                patch_artist=True,
                boxprops=dict(facecolor=colors[idx % len(colors)], color="black"),
                medianprops=dict(color="black"),
                flierprops=dict(markerfacecolor="white", marker=""),
            )

    # cool reference lines
    ax.axhline(y=0.5, color="red", linestyle="-")
    ax.axhline(y=0.5 - competitive_window, color="red", linestyle="--")
    ax.axhline(y=0.5 + competitive_window, color="red", linestyle="--")

    #labeling stuff
    ax.set_xlabel("Ordered Districts")
    ax.set_ylabel("Democrat Vote Share")
    ax.set_title(f"{state} {chamber}: Ordered Seats Plots for " +
                 " and ".join([ensemble_name_dict_for_plots[e] for e in ensemble_list]))
    ax.set_xticks(np.arange(1, num_districts + 1))
    ax.set_xticklabels(np.arange(1, num_districts + 1))

    legend_handles = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=4)
        for i in range(len(ensemble_list))
    ]
    legend_labels = [ensemble_name_dict_for_plots[e] for e in ensemble_list]
    ax.legend(legend_handles, legend_labels, loc="upper left")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

#give a kde stat plot for 2 scores over the ensembles in the list
def kde_jointplot(state, chamber, score1, score2, my_ensemble_list=my_ensemble_list, filename=None, step_size=1,):
    """
    Returns a KDE plot for the scores score1 and score2 over the ensembles in my_ensemble_list.
    Increase step_size to use a subsample and hense speed up the plot.
    """
    # Build the dataframe
    all_rows = []
    for ensemble in my_ensemble_list:
        score_arrays = {
            score: fetch_score_array(state, chamber, ensemble, score)[::step_size] for score in [score1, score2]
        }
        num_plans = len(next(iter(score_arrays.values())))
        for i in range(num_plans):
            row = [score_arrays[score][i] for score in [score1, score2]] + [ensemble]
            all_rows.append(row)
    df = pd.DataFrame(all_rows, columns=[score1, score2, "ensemble"])

    # Create the KDE plot
    ax = sns.kdeplot(df, x=score1, y=score2, hue="ensemble")
    plt.title(f"KDE plot of {score1} vs {score2} for {state} {chamber} ensembles")

    # Fix the legend labels by modifying the existing legend
    if ax.legend_:
        legend = ax.legend_
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        new_labels = [
            ensemble_name_dict_for_plots.get(label, label) for label in labels
        ]

        ax.legend(
            handles=handles, labels=new_labels, title="Ensemble", loc="lower right"
        )

    if filename:
        plt.savefig(filename)
    plt.show()

def plot_mmd_scores(state, chamber, ensemble_list=my_ensemble_list):

    opp_scores = {}
    prop_opp_scores = {}
    coal_scores = {}
    prop_coal_scores = {}

    for ensemble in ensemble_list:
        opp_scores[ensemble] = load_mmd_scores(ensemble, "Opportunity districts")
        prop_opp_scores[ensemble] = load_mmd_scores(ensemble, "Proportional Opportunity")
        coal_scores[ensemble] = load_mmd_scores(ensemble, "Coalition districts")
        prop_coal_scores[ensemble] = load_mmd_scores(ensemble, "Proportional Coalition")

    opp_means = [np.mean(opp_scores[e]) for e in ensemble_list]
    prop_opp_means = [np.mean(prop_opp_scores[e]) for e in ensemble_list]
    
    coal_means = [np.mean(coal_scores[e]) for e in ensemble_list]
    prop_coal_means = [np.mean(prop_coal_scores[e]) for e in ensemble_list]

    x = np.arange(len(ensemble_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, opp_means, width, label="Hispanic opp districts")
    ax.bar(x + width/2, prop_opp_means, width, label="Proportional Hispanic")

    ax.set_xticks(x)
    ax.set_xticklabels([ensemble_name_dict_for_plots[e] for e in ensemble_list], rotation=45)
    ax.set_ylabel("Average number of districts")
    ax.set_title(f"{state} {chamber}: Opportunity vs Proportional Hispanic Districts")
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, coal_means, width, label="Coalition districts")
    ax.bar(x + width/2, prop_coal_means, width, label="Proportional Coalition")

    ax.set_xticks(x)
    ax.set_xticklabels([ensemble_name_dict_for_plots[e] for e in ensemble_list], rotation=45)
    ax.set_ylabel("Average number of districts")
    ax.set_title(f"{state} {chamber}: Coalition vs Proportional Coalition")
    ax.legend()
    plt.tight_layout()
    plt.show()

#displays a heat map for rhat convergence test
def convergence_rhat_plot(state=None, chamber = None, ensembles=list(RECOM_ENSEMBLES.keys()), scores=None, top_chains = None, step_size = 1, rounding = 3):
        if scores is None:
            scores = [
                    "mean-median",
                    "county splits", 
                    "efficiency gap", 
                    "seat bias", 
                    "competitive districts",
                    "Polsbys-Popper", 
                    "Dem seats"
            ]

        rhat_results = {
            "rhat": {}
        }

        for score in scores:
            print(f"\n=== Comparing score: {score} ===")

            array = {ens: fetch_score_array(state, chamber, ens, score)[::step_size] for ens in ensembles}

            for i in range(len(ensembles)):
                for j in range(i + 1, len(ensembles)):
                    e1, e2 = ensembles[i], ensembles[j]
                    a1, a2 = array[e1], array[e2]

                    rhat = gelman_rubin_rhat(a1, a2)    

                    rhat_results["rhat"][(e1, e2, score)] = rhat
                    rhat_results["rhat"][(e2, e1, score)] = rhat

        for score in scores: 
            filtered_rhat = load_filter_rhat(rhat_results, score, top_chains)
            rhat_matrix = make_rhat_matrix(filtered_rhat, top_chains, ensembles)

            colors = [
                (0.0, "#00A65A"),  # 1.00 = perfect green
                (0.7, "#C7E9C0"),  # 1.00–1.05 = good light green
                (0.8, "#FEE391"),  # 1.05–1.10 = warning yellow
                (0.9, "#F8765C"),  # 1.10–1.15 = bad orange
                (1.0, "#B30000"),  # >=1.15 = horrible red
            ]

            rhat_cmap = LinearSegmentedColormap.from_list("rhat_cmap", [c for _, c in colors])

            plt.figure(figsize=(14, 3))
            sns.heatmap(
                    rhat_matrix,
                    annot=True,
                    fmt=".3f",
                    cmap=rhat_cmap,
                    vmin=1,
                    vmax=1.15,
                    linewidths=0.5,
                    linecolor="black"
            )
            plt.title(f"R-hat Convergence Comparison for ({score})")
            plt.tight_layout()
            plt.show()

            
        

                    




    




#functions for creating values tables 

# gives correlation table for scores in list for the ensembles in ensemble list
def correlation_table(state, chamber, my_ensemble_list=my_ensemble_list, my_score_list = scores_output_names,
                       step_size=1, rounding=None, return_dataframe=False,):
    """
    Returns a correlation table for the scores in my_score_list over the ensembles in my_ensemble_list.
    Set step_size to 1 to use all the plans, or a larger number to subsample the data (to use less memory).
    Optionally returns the dataframe used to create the correlation table.
    """
    all_rows = []

    for ensemble in my_ensemble_list:
        score_arrays = {
            score: fetch_score_array(state, chamber, ensemble, score)[::step_size]
            for score in my_score_list
        }
        num_plans = len(next(iter(score_arrays.values())))
        for i in range(num_plans):
            row = [score_arrays[score][i] for score in my_score_list] + [ensemble]
            all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=my_score_list + ["ensemble"])

    corr = df.corr(numeric_only=True)
    if rounding is not None:
        corr = corr.round(rounding)

    return (corr, df) if return_dataframe else corr

#give dataframe for mean-difference between the ensemble and a base ensemble
#maybe don't run this....
def mean_diff_table(score, my_ensemble_list = my_ensemble_list, my_state_chamber_list = state_chamber_list, 
                    pvalue = None, latex_filename = None, rounding = 2, base_column = False):
    """
    Returns a dataframe showing (for each state-chamber pair and each ensemble type)
    the mean-difference between that ensemble and the base0 ensemble with respect to the given score.
    If pvalue is set, it will mark values that are significantly different from the base0 ensemble.
    if latex_filename is set, it will also save the dataframe as a latex table.
    if base_column is True, the table will include a first column that reports the mean of the   ensemble.
    """

    index_list = [f'{a[0]} {a[1]}' for a in my_state_chamber_list] + ['AVERAGE']
    df = pd.DataFrame(columns = my_ensemble_list, index = index_list)
    df_mark = df.copy() # True/False signifying whether the value is marked as statistically significant
    base_col = [] # to hold the base0 means if needed

    for state, chamber in my_state_chamber_list:
        for ensemble in my_ensemble_list:
            mean_diff = mean_diff_dict[(state, chamber, ensemble, score)]
            if score == 'average margin':
                mean_diff = 100 * mean_diff # convert to percentage
            p_value = T_pvalue_dict[(state, chamber, ensemble, score)]
            df.loc[f'{state} {chamber}', ensemble] = mean_diff
            df_mark.loc[f'{state} {chamber}', ensemble] = (pvalue != None and p_value < pvalue)
    if base_column:
        df.insert(0, 'base', base_col + [np.mean(base_col)])
    df.loc['AVERAGE'] = df.mean()
    df = df.applymap(pd.to_numeric)
    df = df.round(rounding)
    df_latex = df.copy()
    df_latex = df_latex.applymap(lambda x: f"{x:.2f}") # round values

    # combine the values and markings into dataframes to return and for Latex
    state_chamber_size_dict = {f'{state} {chamber}': f'{state} {num_seats_dict[(state, chamber)]}' 
                           for state, chamber in my_state_chamber_list}
    for state, chamber in my_state_chamber_list:
        if base_column:
            val = df.loc[f'{state} {chamber}', 'base']
            df_latex.loc[f'{state} {chamber}', 'base'] = f'\\textcolor{{red}}{{ {val:.2f} }}'
        for ensemble in my_ensemble_list:
            val = df.loc[f"{state} {chamber}", ensemble]
            if df_mark.loc[f'{state} {chamber}', ensemble]:
                df_latex.loc[f'{state} {chamber}', ensemble] = f'\\textbf{{{val:.2f}}}'
                df.loc[f'{state} {chamber}', ensemble] = f'*{val:.2f}'
            else:
                df_latex.loc[f'{state} {chamber}', ensemble] = f'{val:.2f}'

    if latex_filename is not None:
        df_latex.rename(columns=ensemble_name_dict_for_plots, index=state_chamber_size_dict, inplace=True)
        df_latex.to_latex(latex_filename, escape=False)
    return df

#give a dataframe that shows distance between the ordered seats plots of the ensembles.
def Ordered_seats_table(my_ensemble_list = my_ensemble_list, combine_method = 'max', seats = 'competitive', rounding = 2, latex_filename = None):
    """
    Returns a dataframe showing (for each state-chamber pair and each ensemble type) 
    the "distance" between the ordered seats plots of the ensemble and the base0 ensemble.
    It sums or averages the ks-distance between the ordered seats.
    If seats == 'competitive', it only considers seats that are competitive for at least one of the two ensembles being compared.
    """
    index_list = [f'{a[0]} {a[1]}' for a in state_chamber_list] + ['AVERAGE']
    df = pd.DataFrame(columns = my_ensemble_list, index = index_list)
    for state, chamber in state_chamber_list:
        for ensemble in my_ensemble_list:
            if seats == 'all':
                if combine_method == 'max':
                    closeness = OSP_all_max[(state, chamber, ensemble)]
                elif combine_method == 'mean':
                    closeness = OSP_all_mean[(state, chamber, ensemble)]
            elif seats == 'competitive':
                if combine_method == 'max':
                    closeness = OSP_competitive_max[(state, chamber, ensemble)]
                elif combine_method == 'mean':
                    closeness = OPS_competitive_mean[(state, chamber, ensemble)]
            df.loc[f'{state} {chamber}', ensemble] = closeness
    df = df.apply(pd.to_numeric)
    df.loc['AVERAGE'] = df.iloc[:-1].mean()
    df = df.round(rounding)

    if latex_filename is not None:
        state_chamber_size_dict = {f'{state} {chamber}': f'{state} {num_seats_dict[(state, chamber)]}' 
                           for state, chamber in state_chamber_list}
        df_latex = df.copy()
        df_latex.rename(columns=ensemble_name_dict_for_plots, index = state_chamber_size_dict, inplace=True)
        df_latex.to_latex(latex_filename, escape=False)







warnings.filterwarnings('ignore')




num_seats_dict = _extract_num_seats(DISTRICTS_BY_STATE)

ordered_seats_plot("CO", "congress", ["co_1-1-1_200000_1", "co_0.6-1-1_200000_1"], competitive_window= 0.05)

for score in scores_output_names:
    kde_plot("CO", "congress", my_ensemble_list, score)

    box_whisker_plot("CO", "congress", my_ensemble_list, score)

kde_jointplot('CO', 'congress', 'Polsby-Popper', 'county splits', my_ensemble_list, filename=None)

kde_jointplot('CO', 'congress', 'mean-median', 'county splits', my_ensemble_list, filename=None)

plot_mmd_scores("CO", "congress", my_ensemble_list)

box_whisker_plots_grid('CO', 'congress', my_ensemble_list, scores_output_names,
                       filename = None , cols=1)



# convergence_test =  ["co_1-1-1_200000_1","co_0.8-1-1_200000_1", "co_0.6-1-1_200000_1"]
# top_chain = ["co_1-1-1_200000_1"]

# convergence_rhat_plot(convergence_test, scores=scores_output_names, top_chains= top_chain)