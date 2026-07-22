import glob
import numpy as np
import pandas as pd 
from pyparsing import results
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

RECOM_ENSEMBLES = {
    "co_1-1-1_200000_1": "Output/co_1-1-1_200000_1/ensemble_1outputs_* (1).csv",
    "co_0.9-1-1_200000_1": "Output/co_0.9-1-1_200000_1/ensemble_1outputs_* (2).csv",
    "co_0.8-1-1_200000_1": "Output/co_0.8-1-1_200000_1/ensemble_1outputs_* (3).csv",
    "co_0.7-1-1_200000_1": "Output/co_0.7-1-1_200000_1/ensemble_1outputs_* (4).csv",
    "co_0.6-1-1_200000_1": "Output/co_0.6-1-1_200000_1/ensemble_1outputs_* (5).csv"
}

def ks_test(a0, a1):
    result = stats.ks_2samp(a0, a1)
    return result.statistic, result.pvalue, result.statistic_sign

def t_test(a0, a1):
    result = stats.ttest_ind(a0, a1, equal_var=False)
    return result.statistic, result.pvalue

def gelman_rubin_rhat(a1, a2):
    n = min(len(a1), len(a2))

    if n < 2:
        return np.nan   

    x1 = a1[:n]
    x2 = a2[:n]

    mu1, mu2 = np.mean(x1), np.mean(x2)
    s1_sq, s2_sq = np.var(x1, ddof=1), np.var(x2, ddof=1)

    W = (s1_sq + s2_sq) / 2
    B = n * ((mu1 - mu2) ** 2) / 2

    if W == 0:
        return 1.0

    V_hat = ((n - 1) / n) * W + (B / n)

    if V_hat < 0:
        V_hat = 0

    R_hat = np.sqrt(V_hat / W)

    return R_hat

def load_recom_scores(folder, score_name):
    files = sorted(glob.glob(folder))

    mapping = {
        "county_splits": "CountySplits",
        "mean-median": "MM",
        "efficiency_gap": "EG",
        "seat bias": "PB",
        "competitive districts": "Comp45-55",
        "Polsby-Popper": "PP",
        "Dem seats": "DWins",
        "average margin": "MM"
    }
     
    col = mapping[score_name]

    arrays = []
    for f in files:
        df = pd.read_csv(f)
        arrays.append(df[col].values)

    return np.concatenate(arrays)

def fetch_score_array(state, chamber, ensemble_type, score):
    
    if ensemble_type in RECOM_ENSEMBLES:
        folder = RECOM_ENSEMBLES[ensemble_type]
        return load_recom_scores(folder, score)

    if score[:3] == "maj":
        a = fetch_score_array(state, chamber, ensemble_type, score[4:])
        if state == "CO" and chamber == "congress":
            return -1 * a
        else:
            return a

def compare_all_ensembles(
    state="CO",
    chamber="congress",
    ensembles=list(RECOM_ENSEMBLES.keys()),
    scores=None,    
    step_size = 1,
    rounding = 3
    ):
        if scores is None:
            scores = [
                    "mean-median",
                    "efficiency_gap", 
                    "seat bias", 
                    "competitive districts",
                    "county_splits", 
                    "Polsby-Popper", 
                    "Dem seats"
            ]
        
        results = {
            "mean_diff": {},
            "ks_stat": {},
            "ks_pvalue": {},
            "t_pvalue": {},
            "rhat": {},
            "kde_plot": {}
        }

        for score in scores:
            print(f"\n=== Comparing score: {score} ===")

            array = {ens: fetch_score_array(state, chamber, ens, score)[::step_size] for ens in ensembles}

            plt.figure(figsize=(10, 6))
            for ens in ensembles:
                sns.kdeplot(array[ens], label=ens)
            plt.title(f"KDE Comparison for {score}")
            plt.legend()
            plt.tight_layout()
            results["kde_plot"][score] = plt.gcf()
            plt.show()

            for i in range(len(ensembles)):
                for j in range(i + 1, len(ensembles)):
                    e1, e2 = ensembles[i], ensembles[j]
                    a1, a2 = array[e1], array[e2]

                    mean_diff = np.mean(a2) - np.mean(a1)
                    ks_stat, ks_pvalue, ks_sign = ks_test(a1, a2)
                    _, t_pvalue = t_test(a1, a2)
                    rhat = gelman_rubin_rhat(a1, a2)    

                    results["mean_diff"][(e1, e2, score)] = round(mean_diff, rounding)
                    results["ks_stat"][(e1, e2, score)] = round(ks_stat * ks_sign, rounding)
                    results["ks_pvalue"][(e1, e2, score)] = ks_pvalue
                    results["t_pvalue"][(e1, e2, score)] = t_pvalue

                    results["rhat"][(e1, e2, score)] = rhat
                    results["rhat"][(e2, e1, score)] = rhat

        return results
  
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

def display_rhat_heatmap(rhat_matrix, score_name):
    plt.figure(figsize=(14, 3))
    sns.heatmap(
        rhat_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=1.0,
        vmax=max(1.1, rhat_matrix.max().max()),
        linewidths=0.5,
        linecolor="black"
    )
    plt.title(f"R-hat Comparison of Top Chains vs All Ensembles ({score_name})")
    plt.tight_layout()
    plt.show()

def filter_rhat_for_top_chains(compare_results, score, top_chains):
    return {
        key: value
        for key, value in compare_results["rhat"].items()
        if key[2] == score and (key[0] in top_chains or key[1] in top_chains)
    }

compare_results = compare_all_ensembles(
    state = "CO",
    chamber = "congress",
    ensembles=list(RECOM_ENSEMBLES.keys()),
    scores=[
        "mean-median",
        "efficiency_gap", 
        "seat bias", 
        "competitive districts",
        "county_splits", 
        "Polsby-Popper", 
        "Dem seats"
    ],
    step_size = 1,
    rounding = 3
)
 
top_chains = ["co_1-1-1_200000_1"]
scores_to_plot = ["mean-median", "seat bias", "county_splits", "Polsby-Popper", "Dem seats"]
ensembles = list(RECOM_ENSEMBLES.keys())

for score in scores_to_plot:

    filtered_rhat = filter_rhat_for_top_chains(compare_results, score, top_chains)
    rhat_matrix = make_rhat_matrix(filtered_rhat, top_chains, ensembles)
    display_rhat_heatmap(rhat_matrix, score)



# ≈ 1.00  converged

# 1.00–1.05  probably converged

# > 1.05  not converged

# > 1.10  definitely not converged
