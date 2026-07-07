from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)

from gerrychain.proposals import recom, propose_random_flip

from gerrychain.tree import recursive_tree_part, recursive_seed_part, bipartition_tree

from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, partisan_bias

from gerrychain.updaters import cut_edges

from gerrychain.tree import bipartition_tree, find_balanced_edge_cuts_memoization


import geopandas as gpd
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from functools import partial 

import csv
import pandas as pd

import json

graph = Graph.from_json("state_data/ma/ma.json")
df = gpd.read_file("state_data/ma/ma.shp")

ideal_population = df['TOTPOP'].sum()/7
#plt.figure(figsize=(14,10))
#nx.draw(graph, pos = {x:(graph.nodes()[x]['C_X'],graph.nodes()[x]['C_Y']) for x in graph.nodes()},node_color=[graph.nodes()[x]['CON'] for x in graph.nodes()],cmap='tab20b',node_size=15)

def count_spanning(graph):
    laplacian = nx.laplacian_matrix(graph)
    L = np.delete(np.delete(laplacian.todense(), 0, 0), 1, 1)
    return np.linalg.slogdet(L)[1]

def county_splits(partition, df=df):
    df["current"] = df["PRECINCT20"].map(partition.assignment)

    counties = sum(df.groupby("COUNTY")['current'].nunique()>1)
    return counties

election_names = [
    "PRE"
]

num_elections = len(election_names)

election_columns = [
['G24PRERTRU','G24PREDHAR']
]

my_updaters = {
    "population": updaters.Tally("population", alias="population"),
    "cut_edges": cut_edges,
    "PP":polsby_popper,
    "county_splits": county_splits
}

elections = [
    Election(
        election_names[i],
        {"Democratic": election_columns[i][1], "Republican": election_columns[i][0]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}

for node in graph.nodes():
    graph.nodes()[node]["non_NH_Black"] = graph.nodes()[node]["TOTPOP"] - graph.nodes()[node]["NH_BLACK"]
    graph.nodes()[node]["non_Hispanic"] = graph.nodes()[node]["TOTPOP"] - graph.nodes()[node]["HISP"]

my_updaters.update({"NH_Black":Election("NH_Black",{"NH_Black": "NH_Black", "non_NH_Black": "non_NH_Black"})})
my_updaters.update({"Hispanic":Election("Hispanic",{"Hispanic": "Hispanic", "non_Hispanic": "non_Hispanic"})})

# save percentages

my_updaters.update(election_updaters)

CON_Part = GeographicPartition(graph,"CD",my_updaters)

print(CON_Part['county_splits'])
print(CON_Part['population'])
print(sum([1/x for x in polsby_popper(CON_Part).values()])/7)
print([(x-ideal_population)/ideal_population for x in CON_Part['population'].values()])
print(sorted(CON_Part['PRE'].percents("Democratic")))
print(sorted(CON_Part['NH_Black'].percents("NH_Black")))