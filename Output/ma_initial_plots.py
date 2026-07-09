import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

max_steps =120_000
step_size = 10_000
ts = [x * step_size for x in range(1, int(max_steps / step_size) + 1)]

df_1st = pd.read_csv("/Users/alexandrarossi/Desktop/STEMinists/Output/Testing_03/Testing_03chain_outputs_10000.csv")
df_2nd = pd.read_csv("/Users/alexandrarossi/Desktop/STEMinists/Output/Testing_04/Testing_04chain_outputs_10000.csv")

for t in ts[1:]: 

    df2 = pd.read_csv(f"/Users/alexandrarossi/Desktop/STEMinists/Output/Testing_03/Testing_03chain_outputs_{t}.csv")
    df_1st = pd.concat([df_1st,df2],ignore_index=True)

for t in ts[1:]:
    df_2nd_2 = pd.read_csv(f"/Users/alexandrarossi/Desktop/STEMinists/Output/Testing_04/Testing_04chain_outputs_{t}.csv")
    df_2nd = pd.concat([df_2nd,df_2nd_2],ignore_index=True)

dfs = [df_1st, df_2nd]

for df in dfs:

    counts = Counter(df['CountySplits'])

    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.axvline(9, color='red',label='Enacted')
    plt.title("County splits")
    plt.show()

    plt.hist(df['MM'],bins=100)
    plt.axvline(0, color='red',label='Enacted')
    plt.title("Mean median")
    plt.show()

    plt.hist(df['EG'],bins=100)
    plt.axvline(0.152, color='red',label='Enacted')
    plt.title("Efficiency gap")
    plt.show()

    #plt.hist(df['PB'])

    counts = Counter(df['PB'])
    categories = list(counts.keys())
    frequencies = list(counts.values())
    plt.bar(categories, frequencies)
    plt.axvline(0, color='red',label='Enacted')
    plt.title("Partisan bias")
    plt.show()


    plt.hist(df['PP'])
    plt.axvline(22.3, color='red',label='Enacted')
    plt.title("Polsby popper")
    plt.show()

    counts = Counter(df['DWins'])

    categories = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(categories, frequencies)
    plt.axvline(9, color='red',label='Enacted')
    plt.title("Dem wins")
    plt.show()


    counts = Counter(df['Comp45-55'])

    categories = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(categories, frequencies)
    plt.axvline(0, color='red',label='Enacted')
    plt.title("Competitive Districts")
    plt.show()