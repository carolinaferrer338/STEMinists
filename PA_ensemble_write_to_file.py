from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)

from gerrychain.proposals import recom, propose_random_flip

from gerrychain.tree import recursive_tree_part, recursive_seed_part, bipartition_tree

from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, partisan_bias

from gerrychain.updaters import cut_edges

from gerrychain.tree import bipartition_tree, find_balanced_edge_cuts_memoization

from pathlib import Path
import time
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt

import networkx as nx
from functools import partial 

import csv
import pandas as pd

import json
import os
import sys
import random
import numpy as np
import pickle as pkl

from tqdm import tqdm

#steps and proposals 
MAIN_CHAIN_STEPS = 200_000
TREE_PROPOSAL_RETRIES = 10_000

#state informaiton
STATE_ABBR = sys.argv[1]

#alpha value settings
COMPETITIVENESS_ALPHA = float(sys.argv[2])
CUT_EDGES_ALPHA = float(sys.argv[3])
COUNTY_SPLITS_ALPHA = float(sys.argv[4])

INITIAL_COUNTY_SPLITS = {
    'pa': 13, 
    'ma': 9,
    'ut': 10,
    'tn': 20, 
    'co': 13
}[STATE_ABBR]

COUNTY_FIELD_NAME = {
    'pa': "COUNTYFP20", 
    'ma': "COUNTY",
    'ut': "CountyID20",
    'tn': "COUNTYFP18", 
    'co': "COUNTYFP20"
}[STATE_ABBR]

N_CONG_DISTS = {
    'pa': 17, 
    'ma': 9,
    'ut': 4,
    'tn': 9, 
    'co': 8
}[STATE_ABBR]

#file names and directory
OUTPUT_DIR = f"Output/{STATE_ABBR}_{COMPETITIVENESS_ALPHA}-{CUT_EDGES_ALPHA}-{COUNTY_SPLITS_ALPHA}_{MAIN_CHAIN_STEPS}"

graph = Graph.from_json(f"state_data/{STATE_ABBR}/{STATE_ABBR}.json")
df = gpd.read_file(f"state_data/{STATE_ABBR}/{STATE_ABBR}.shp")



ideal_population = df['TOTPOP'].sum()/N_CONG_DISTS
df['C_X'] = df.centroid.x
df['C_Y'] = df.centroid.y

graph.add_data(df, columns=['C_X', 'C_Y'])

#plots discreet graph 
plt.figure(figsize=(14,10))
node_color_str = [graph.nodes()[x]['CD'] for x in graph.nodes()]
node_color_int= []
for i in node_color_str: 
    node_color_int.append(int(i))
nx.draw(graph, pos = {x:(graph.nodes()[x]['C_X'],graph.nodes()[x]['C_Y']) for x in graph.nodes()},node_color=node_color_int,
        cmap='tab20b',node_size=15)
    
#plots congressional districts 
df.plot(column="CD",cmap='tab20b')
plt.axis('off')



#updaters 
#i added this for mmd
for node in graph.nodes():
    graph.nodes()[node]["TOTPOP"] = graph.nodes()[node]["TOTPOP"]
    graph.nodes()[node]["BLACK"] = graph.nodes()[node]["NH_BLACK"]
    graph.nodes()[node]["HISP"] = graph.nodes()[node]["HISP"]
    graph.nodes()[node]["WHITE"] = graph.nodes()[node]["NH_WHITE"]

def calc_minority_metrics(statewide_demos, demos_by_dist):
    minority_groups = ["NH_BLACK", "HISP"]

    statewide_total = statewide_demos["TOTPOP"]
    statewide_minority_total = sum(statewide_demos[g] for g in minority_groups)

    statewide_minority_share = statewide_minority_total / statewide_total

    opportunity_districts = 0
    coalition_districts = 0

    for dist in demos_by_dist:
        total = dist["TOTPOP"]
        if total == 0:
            continue

        # Single-group opportunity (any minority > 50%)
        for g in minority_groups:
            if dist[g] / total >= 0.50:
                opportunity_districts += 1
                break

        combined_minority = sum(dist[g] for g in minority_groups)
        if combined_minority / total >= 0.50:
            coalition_districts += 1

    num_districts = len(demos_by_dist)
    proportional_opportunities = statewide_minority_share * num_districts
    proportional_coalitions = proportional_opportunities  # same formula

    return {
        "opportunity_districts": opportunity_districts,
        "coalition_districts": coalition_districts,
        "proportional_opportunities": proportional_opportunities,
        "proportional_coalitions": proportional_coalitions
    }

def get_statewide_demos(graph):
    demo_cols = ["NH_WHITE","NH_BLACK","HISP","TOTPOP"]
    return {col: sum(graph.nodes[n].get(col,0) for n in graph.nodes()) for col in demo_cols}

def get_demos_by_district(partition):
    demo_cols = ["NH_WHITE","NH_BLACK","HISP","TOTPOP"]
    demos = []
    for dist in sorted(partition.parts.keys()):
        nodes = partition.parts[dist]
        totals = {col: sum(partition.graph.nodes[n].get(col,0) for n in nodes)
                  for col in demo_cols}
        demos.append(totals)
    return demos

statewide_demos = get_statewide_demos(graph)
#end of added stuff

def count_spanning(graph):
    laplacian = nx.laplacian_matrix(graph)
    L = np.delete(np.delete(laplacian.todense(), 0, 0), 1, 1)
    return np.linalg.slogdet(L)[1]

def county_splits(partition, df=df):
    df["current"] = df.index.map(partition.assignment)
    counties = sum(df.groupby(COUNTY_FIELD_NAME)['current'].nunique()>1)
    return counties

def comp_dist(partition):
    return sum([abs(x-.5)<.05 for x in partition['PRE20'].percents("Democratic")])

def avg_pp(partition):
    return sum([x for x in polsby_popper(partition).values()])/N_CONG_DISTS

def least_democratic(partition):
    return min(partition['PRE20'].percents("Democratic"))

election_names = [
    "PRE20"
]

num_elections = len(election_names)

election_columns = [
['PRE20R','PRE20D']
]

my_updaters = {
    "population": updaters.Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
    "PP":polsby_popper,
    "county_splits": county_splits,
    "comp_dists": comp_dist,
    "least_demo": least_democratic,
    "avg_pp": avg_pp
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
    graph.nodes()[node]["non_NH_BLACK"] = graph.nodes()[node]["TOTPOP"] - graph.nodes()[node]["NH_BLACK"]
    graph.nodes()[node]["non_HISP"] = graph.nodes()[node]["TOTPOP"] - graph.nodes()[node]["HISP"]

my_updaters.update({"NH_BLACK":Election("NH_BLACK",{"NH_BLACK": "NH_BLACK", "non_NH_BLACK": "non_NH_BLACK"})})
my_updaters.update({"HISP":Election("HISP",{"HISP": "HISP", "non_HISP": "non_HISP"})})

# save percentages
my_updaters.update(election_updaters)

# enacted plan
enacted_plan = Partition(graph,
                           df["CD"],
                           my_updaters)

#checking if its working 
CON_Part = GeographicPartition(graph,"CD",my_updaters)

print(CON_Part['county_splits'])
print(CON_Part['population'])
print(CON_Part['comp_dists'])
print(CON_Part['least_demo'])
print(sum([1/x for x in polsby_popper(CON_Part).values()])/N_CONG_DISTS)
print([(x-ideal_population)/ideal_population for x in CON_Part['population'].values()])
print(sorted(CON_Part['PRE20'].percents("Democratic")))
print(sorted(CON_Part['NH_BLACK'].percents("NH_BLACK")))

#competativeness contraint for accepting 
def accept_closer_competitive(partition):

    if partition['comp_dists'] > partition.parent['comp_dists']:
        return True
    
    if partition['comp_dists'] < partition.parent['comp_dists']:
        return False
    
    closeness_value_new = min([abs(x-.5) for x in partition["PRE20"].percents("Democratic") if abs(x-.5)>.05])
    closeness_value_old = min([abs(x-.5) for x in partition.parent["PRE20"].percents("Democratic") if abs(x-.5)>.05])

    if closeness_value_new <= closeness_value_old:
        return True
    if closeness_value_new > closeness_value_old:
        alpha = random.random()
        if alpha < COMPETITIVENESS_ALPHA:
            return False
        else:
            return True

#competativeness contraint for accepting 
def accept_higher_pp(partition):

    if partition['avg_pp'] > partition.parent['avg_pp']:
        return True
    
    if partition['avg_pp'] < partition.parent['avg_pp']:
        return False
    
    avg_pp_new = sum([1/x for x in polsby_popper(partition).values()])/N_CONG_DISTS
    avg_pp_old = sum([1/x for x in polsby_popper(partition.parent).values()])/N_CONG_DISTS

    if avg_pp_new >= avg_pp_old:
        return True
    if avg_pp_new < avg_pp_old:
        alpha = random.random()
        if alpha < 0.99:
            return False
        else:
            return True

def accept_lower_ces(partition):
    
    if len(partition['cut_edges']) <= len(partition.parent['cut_edges']):
        return True
    
    if len(partition['cut_edges']) > len(partition.parent['cut_edges']):
        alpha = random.random()
        if alpha < CUT_EDGES_ALPHA:
            return False
        else: 
            return True
        
'''
def accept_lower_splt(partition):
    if partition['county_splits'] <= partition.parent['county_splits']:
        return True
    
    if partition['county_splits'] > partition.parent['county_splits']:
        alpha = random.random()
        if alpha < COUNTY_SPLITS_ALPHA:
            return False
        else: 
            return True
'''

def combined_acceptance(partition):
    # takes in a partition
    
    comp = accept_closer_competitive(partition)

    ces = accept_lower_ces(partition)
    
    # sp = accept_lower_splt(partition)
    
    total = ces + comp
    if total < 2:
        return False
    else: 
        return True

def county_constraint(partition):
    return partition['county_splits'] < 50

def pp_constraint(partition): 

    return sum([1/x for x in polsby_popper(partition).values()])/N_CONG_DISTS > 3

ces_constraint = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(enacted_plan["cut_edges"])
)

def competitiveness_constraint(partition):

    return sum([abs(x-.5)<.05 for x in partition['PRE20'].percents("Democratic")]) > -1

#starting with a seed
def create_init_state():
    cd_dict =  recursive_tree_part(graph,range(N_CONG_DISTS),ideal_population,'TOTPOP', epsilon = 0.02)

    tree_partition = GeographicPartition(graph,cd_dict,my_updaters)

    plt.plot(figsize=[14,10])
    nx.draw(graph, pos = {x:(graph.nodes()[x]['C_X'],graph.nodes()[x]['C_Y']) for x in graph.nodes()},node_color=[cd_dict[x] for x in graph.nodes()],
            cmap='tab20',node_size=15)

    #creating Markov chains 
    print(f"The initial tree seed splits {tree_partition['county_splits']} counties.")
        
    county_proposal = partial(
        recom,
        pop_col = "TOTPOP",
        pop_target=ideal_population,
        epsilon=0.02,
        node_repeats=2,
        region_surcharge = {COUNTY_FIELD_NAME:COUNTY_SPLITS_ALPHA},
        method = partial(bipartition_tree,max_attempts= TREE_PROPOSAL_RETRIES,  warn_attempts = 1000,  allow_pair_reselection = True)
    )

    initial_county_chain = MarkovChain(
        proposal=county_proposal,
        constraints=[],
        accept=accept.always_accept,
        initial_state=tree_partition,
        total_steps=1000
    )

    for part in tqdm(initial_county_chain):       
        if part['county_splits'] < INITIAL_COUNTY_SPLITS:
            break   

    initial_competitiveness_chain = MarkovChain(
        proposal=county_proposal,
        constraints= county_constraint,
        accept=accept_closer_competitive,
        initial_state=part,
        total_steps=10_000
    )
    
    cds = []
           
    for part in tqdm(initial_competitiveness_chain):
        cds.append(min([abs(x-.5) for x in part["PRE20"].percents("Democratic") if abs(x-.5)>.05]))            
        if part['comp_dists'] > 0:
            break

    fourth_recom_chain = MarkovChain(
        proposal=county_proposal,
        constraints= [county_constraint, competitiveness_constraint],
        accept=accept_higher_pp,
        initial_state=part,
        total_steps=10_000
    )

    avg_pps = []

    for part in tqdm(fourth_recom_chain):
        avg_pps.append(sum([x for x in polsby_popper(part).values()])/N_CONG_DISTS)
        print(avg_pps[-1])
            
        if pp_constraint(part) == True:
            break

    new_starting_seed = GeographicPartition(graph, dict(part.assignment), my_updaters)
    print(f"The new tree seed splits {new_starting_seed['county_splits']} counties.")
    return new_starting_seed

first_seed = create_init_state()
second_seed = create_init_state()

#contraints and proposal
county_proposal = partial(
    recom,
    pop_col="TOTPOP",
    pop_target=ideal_population,
    epsilon=0.02,
    node_repeats=2,
    region_surcharge = {COUNTY_FIELD_NAME:COUNTY_SPLITS_ALPHA},
    method = partial(bipartition_tree,max_attempts= 10000,  warn_attempts = 1000,  allow_pair_reselection = True)
)

Path(f"{OUTPUT_DIR}_1/").mkdir(parents=True, exist_ok=True)

Path(f"{OUTPUT_DIR}_2/").mkdir(parents=True, exist_ok=True)

#markov chain definition and calling it to run
#also writes stuff to file
def run_markov_chain(seed, proposal_function, constraint_choices, file_name, accept_function, num_steps=100_000):

    second_recom_chain = MarkovChain(
        proposal=proposal_function,
        constraints=constraint_choices,
        accept=accept_function,
        initial_state=seed,
        total_steps=num_steps
    )

    cs = []
    mms = []
    egs = []
    pbs =[]
    dvp = []
    pps = []
    bvp = []
    mbvp = []
    wins = []
    cds = []
    ces = []

    #i added this
    opp_scores = []
    coal_scores = []
    prop_opp_scores = []
    prop_coal_scores = []
    #end of what i added

    pbar = tqdm(total=num_steps)
    for temp, part in tqdm(enumerate(second_recom_chain)):
        pbar.update(1)
        if temp % 10_000 == 0:
        
            ad = dict(part.assignment)

            with open(f"{file_name}_{temp}.json", "w") as file:
                json.dump(ad, file)

            plt.figure(figsize=(14,10))
            nx.draw(graph, pos = {x:(graph.nodes()[x]['C_X'],graph.nodes()[x]['C_Y']) for x in graph.nodes()},node_color=[ad[x] for x in graph.nodes()],
                cmap='tab20b',node_size=15)
            plt.savefig(f'./{file_name}network_plot_{temp}.png')
            plt.close()
    
            df['current'] = df.index.map(ad)
            df.plot(column='current',cmap='tab20b')
            plt.axis('off')
            plt.savefig(f'./{file_name}df_plot_{temp}.png')
            plt.close()


            ndf = pd.DataFrame({"CountySplits":cs, "MM":mms, 'EG':egs,'PB':pbs,'DWins':wins,'PP':pps,'Comp45-55':cds, 'CES'; ces})

            mmd = pd.DataFrame({"Opportunity districts":opp_scores, "Coalition districts":coal_scores, 'Proportional Opportunity':prop_opp_scores,'Proportional Coalition':prop_coal_scores})

            ndf.to_csv(f"./{file_name}chain_outputs_{temp}.csv")

            mmd.to_csv(f"./{file_name}mmd_outputs_{temp}.csv")

            with open(f"./{file_name}_DemPercs_{temp}.csv", "w") as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(dvp)
        
        
            with open(f"./{file_name}_BlackPercs_{temp}.csv", "w") as tf1:
                writer = csv.writer(tf1, lineterminator="\n")
                writer.writerows(bvp)

            cs = []
            mms = []
            egs = []
            pbs =[]
            dvp = []
            pps = []
            bvp = []
            mbvp = []
            wins = []
            cds = []
            ces = []

            #I added thisisss
            opp_scores = []
            coal_scores = []
            prop_opp_scores = []
            prop_coal_scores = []
            #end of stuff added
    
        cs.append(part['county_splits'])
        mms.append(mean_median(part['PRE20']))
        egs.append(efficiency_gap(part['PRE20']))
        pbs.append(partisan_bias(part['PRE20']))
        dvp.append(sorted(part['PRE20'].percents("Democratic")))
        pps.append(sum([1/x for x in polsby_popper(part).values()])/N_CONG_DISTS)
        bvp.append(sorted(part['NH_BLACK'].percents("NH_BLACK")))
        mbvp.append(max(bvp[-1]))
        wins.append(part['PRE20'].wins("Democratic"))
        cds.append(sum([abs(x-.5)<.05 for x in part['PRE20'].percents("Democratic")]))
        ces.append(part['cut_edges'])

    ##i added this for mmd
        demos_by_district = get_demos_by_district(part)
        minority_scores = calc_minority_metrics(statewide_demos, demos_by_district)

        opp_scores.append(minority_scores["opportunity_districts"])
        coal_scores.append(minority_scores["coalition_districts"])
        prop_opp_scores.append(minority_scores["proportional_opportunities"])
        prop_coal_scores.append(minority_scores["proportional_coalitions"])

    #end of stuff addedd
print("Starting at", datetime.fromtimestamp(time.time()))
run_markov_chain(first_seed, county_proposal, [ces_constraint, competitiveness_constraint, county_constraint], f"{OUTPUT_DIR}_1/ensemble_1", combined_acceptance, num_steps=MAIN_CHAIN_STEPS)
print(f"First chain of {OUTPUT_DIR} done at", datetime.fromtimestamp(time.time()))
run_markov_chain(second_seed, county_proposal, [ces_constraint, competitiveness_constraint, county_constraint], f"{OUTPUT_DIR}_2/ensemble_2", combined_acceptance, num_steps=MAIN_CHAIN_STEPS)
print(f"Second chain of {OUTPUT_DIR} done at", datetime.fromtimestamp(time.time()))
