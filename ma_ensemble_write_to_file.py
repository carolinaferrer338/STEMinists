from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)

from gerrychain.proposals import recom, propose_random_flip

from gerrychain.tree import recursive_tree_part, recursive_seed_part, bipartition_tree

from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, partisan_bias

from gerrychain.updaters import cut_edges

from gerrychain.tree import bipartition_tree, find_balanced_edge_cuts_memoization

from pathlib import Path
import time

import geopandas as gpd
import matplotlib.pyplot as plt

import networkx as nx
from functools import partial 

import csv
import pandas as pd

import json
import random
import numpy as np

#graph and reads files 
graph = Graph.from_json("state_data/ma/ma.json")
df = gpd.read_file("state_data/ma/ma.shp")
n = 9

ideal_population = df['TOTPOP'].sum()/n
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

    counties = sum(df.groupby("COUNTY")['current'].nunique()>1)
    return counties

def comp_dist(partition):
    return sum([abs(x-.5)<.05 for x in partition['PRE20'].percents("Democratic")])

def avg_pp(partition):
    return sum([x for x in polsby_popper(partition).values()])/n

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
print(sum([1/x for x in polsby_popper(CON_Part).values()])/n)
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
        if alpha < 0.5:
            return False
        else:
            return True

#competativeness contraint for accepting 

def accept_lower_ces(partition):
    
    if len(partition['cut_edges']) <= len(partition.parent['cut_edges']):
        return True
    
    if len(partition['cut_edges']) > len(partition.parent['cut_edges']):
        alpha = random.random()
        if alpha < 0.5:
            return False
        else: 
            return True
        

def accept_lower_splt(partition):
    if partition['county_splits'] <= partition.parent['county_splits']:
        return True
    
    if partition['county_splits'] > partition.parent['county_splits']:
        alpha = random.random()
        if alpha < 0.5:
            return False
        else: 
            return True


def combined_acceptance(partition):
    # takes in a partition
    
    comp = accept_closer_competitive(partition)

    ces = accept_lower_ces(partition)
    
    splt = accept_lower_splt(partition)
    
    total = splt + ces + comp
    if total < 3:
        return False
    else: 
        return True

def county_constraint(partition):
    return partition['county_splits'] < 11

def pp_constraint(partition): 

    return sum([1/x for x in polsby_popper(partition).values()])/n > 3 #not using

ces_constraint = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(enacted_plan["cut_edges"])
)

def competitiveness_constraint(partition):

    return sum([abs(x-.5)<.05 for x in partition['PRE20'].percents("Democratic")]) > 0
#starting with a seed
def create_init_state():
    cd_dict =  recursive_tree_part(graph,range(n),ideal_population,'TOTPOP', epsilon = 0.02)

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
        region_surcharge = {"COUNTY":1},
        method = partial(bipartition_tree,max_attempts= 10000,  warn_attempts = 1000,  allow_pair_reselection = True)
    )

    second_recom_chain = MarkovChain(
        proposal=county_proposal,
        constraints=[],
        accept=accept.always_accept,
        initial_state=tree_partition,
        total_steps=1000
    )

    temp = 0
    for part in second_recom_chain:
        temp +=1
        if temp %10 == 0:
            print(temp)
            
        if part['county_splits'] < 13:
            break   

    third_recom_chain = MarkovChain(
        proposal=county_proposal,
        constraints= county_constraint,
        accept=accept_closer_competitive,
        initial_state=part,
        total_steps=100_000
    )
    temp = 0
    cds = []
    for part in third_recom_chain:
        temp +=1
        cds.append(min([abs(x-.5) for x in part["PRE20"].percents("Democratic") if abs(x-.5)>.05]))
        print(cds[-1])
        if temp %10 == 0:
            print(temp)
            
        if part['comp_dists'] > 0:
            break

    fourth_recom_chain = MarkovChain(
        proposal=county_proposal,
        constraints= [county_constraint, competitiveness_constraint],
        accept=accept_lower_ces,
        initial_state=part,
        total_steps=10_000
    )

    temp = 0
    ces_ss = []
    for part in fourth_recom_chain:
        temp +=1
        ces_ss.append(len(part['cut_edges']))
        print(ces_ss[-1])
        if temp %10 == 0:
            print(temp)
            
        if ces_constraint == True:
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
    epsilon=0.01,
    node_repeats=2,
    region_surcharge = {"COUNTY":1},
    method = partial(bipartition_tree,max_attempts= 10000,  warn_attempts = 1000,  allow_pair_reselection = True)
)
Path("Output/PA_Testing_01/").mkdir(parents=True, exist_ok=True)

Path("Output/PA_Testing_02/").mkdir(parents=True, exist_ok=True)

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

    #i added this
    opp_scores = []
    coal_scores = []
    prop_opp_scores = []
    prop_coal_scores = []
    #end of what i added

    temp = 0

    for part in second_recom_chain:

        temp += 1

        if temp %10_000 == 0:
        
            print("Step Number", temp)
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


            ndf = pd.DataFrame({"CountySplits":cs, "MM":mms, 'EG':egs,'PB':pbs,'DWins':wins,'PP':pps,'Comp45-55':cds})

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
        pps.append(sum([1/x for x in polsby_popper(part).values()])/n)
        bvp.append(sorted(part['NH_BLACK'].percents("NH_BLACK")))
        mbvp.append(max(bvp[-1]))
        wins.append(part['PRE20'].wins("Democratic"))
        cds.append(sum([abs(x-.5)<.05 for x in part['PRE20'].percents("Democratic")]))


    ##i added this for mmd
        demos_by_district = get_demos_by_district(part)
        minority_scores = calc_minority_metrics(statewide_demos, demos_by_district)

        opp_scores.append(minority_scores["opportunity_districts"])
        coal_scores.append(minority_scores["coalition_districts"])
        prop_opp_scores.append(minority_scores["proportional_opportunities"])
        prop_coal_scores.append(minority_scores["proportional_coalitions"])

    #end of stuff addedd
print("Starting at", time.time())
run_markov_chain(first_seed, county_proposal, [ces_constraint, competitiveness_constraint, county_constraint], "Output/PA_Testing_01/Testing_01", combined_acceptance, num_steps=100)
print("First chain done at", time.time())
run_markov_chain(second_seed, county_proposal, [ces_constraint, competitiveness_constraint, county_constraint], "Output/PA_Testing_02/Testing_02", combined_acceptance, num_steps=100)
print("Second chain done at", time.time())
