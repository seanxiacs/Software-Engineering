import matplotlib.pyplot as plt
from gerrychain import (Partition, Graph, MarkovChain,
                        updaters, constraints, accept,
                        GeographicPartition)
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.constraints import contiguous
from functools import partial
import pandas
import geopandas as gpd
from gerrychain import Election
# from pcompress import Record
import pcompress
import tqdm
import time  # Import the time module
from gerrychain.updaters import Tally, cut_edges
from gerrychain import metrics
import seaborn as sns

# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # If you wish to see progress

import pandas as pd
# import seaborn as sns
import json

# Define the number of plans you want in the ensemble
# number_of_plans = 5_000  # Change this to 250 or 5_000 as needed
state = "SC"
number_of_plans = 10
population_epsilon = 0.45 #for sc, 0.45 is the epsilon value
# population_epsilon = 0.25 #for ny, 0.25 is the epsilon value
node_repetitions = 1_000
# Define racial groups to process
racial_groups = ['white', 'black', 'asian', 'hispanic']
print("The racial groups are:", racial_groups)

# Start timing the file loading
start_time_file = time.time()

graph = Graph.from_json("sc_trimmed_graph.json")
print(f"Working with state: {state} and number of plans: {number_of_plans} with node repetitions: {node_repetitions}.")

# Stop timing the file loading
end_time_file = time.time()

# Calculate and print the time taken to read the file
time_taken_file = end_time_file - start_time_file
print(f"Time taken to load the file: {time_taken_file} seconds")

my_updaters = {
    "population": updaters.Tally("Total", alias="population"),
    "white": updaters.Tally("White", alias="white"),
    "black": updaters.Tally("Black", alias="black"),
    "asian": updaters.Tally("Asian", alias="asian"),
    "hispanic": updaters.Tally("Hispanic/Latino", alias="hispanic"),
    "cut_edges": updaters.cut_edges,
    # "perimeter": updaters.perimeter,
    # "area": updaters.Tally("area", alias="area"),
}

elections = [
    Election("PRES20", {"Democratic": "Biden Votes", "Republican": "Trump Votes"}),
]

election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = Partition(
    graph,
    assignment="districtID",
    updaters=my_updaters
)

# initial_partition.plot()

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

proposal = partial(
    recom,
    pop_col="Total",
    pop_target=ideal_population,
    epsilon=population_epsilon,
    node_repeats=node_repetitions,
)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, population_epsilon)

recom_chain = MarkovChain(
    proposal=proposal,
    constraints=[
        contiguous,
        pop_constraint,
        compactness_bound,
        ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=number_of_plans,
)

assignment_list = []

# SeaWulf-4. Calculate election winners (required) (AD) 
def calculate_district_winners(partition):
    election_data = partition["PRES20"]  # Accessing the election data
    dem_votes = election_data.counts("Democratic")  # Democratic votes per district
    rep_votes = election_data.counts("Republican")  # Republican votes per district
    
    winners = [] # Initialize a list to store the winners for each district
    
    # Determine the winner for each district and add to the list
    for dem, rep in zip(dem_votes, rep_votes):
        winner = "D" if dem > rep else "R"
        winners.append(winner)
    
    return tuple(winners) # Convert the list of winners to a tuple before returning

#We can use the results of the caculate_election_splits in order to calculate the winners
def calculate_district_winners_using_election_splits(split_results):
    dem_votes = split_results["Democratic"]
    rep_votes = split_results["Republican"]
    
    winners = [] # Initialize a list to store the winners for each district
    
    # Determine the winner for each district and add to the list
    for dem, rep in zip(dem_votes, rep_votes):
        winner = "D" if dem > rep else "R"
        winners.append(winner)
    
    return tuple(winners) # Convert the list of winners to a tuple before returning

# SeaWulf-5. Calculate the Republican/Democratic split for each random district plan (required) (AD)
def calculate_election_splits(partition):
    results = {}
    election_data = partition["PRES20"]
    dem_votes = election_data.counts("Democratic")
    rep_votes = election_data.counts("Republican")
    results = {"Democratic": dem_votes, "Republican": rep_votes}
    return results
    # return partition["PRES16"].totals

# SeaWulf-6. Identify and store additional random district plans of note (required) (AD) 
maximum_democratic_winners = 0
maximum_republican_winners = 0
minimum_democratic_winners = 200
minimum_republican_winners = 200

def identify_and_store_district_plans(partition, path):
    #Implement later
    # pass
    
    # partition.to_json("scnewplan.json")
    # partition.graph.to_json("scnewplan.json")
    
    # partition.graph.to_json(path + ".json")
    return True

# SeaWulf-7. Calculate ensemble measures (required) (AD)
total_democratic_district_wins = 0
total_republican_district_wins = 0
# Moved to the bottom under the markov chain loop (recom)

def calculate_racial_percentages(partition, race):
    # Extract values from the partition's updaters
    race_values = list(partition[race].values())
    population_values = list(partition["population"].values())

    # Calculate the percentage of white population relative to total population per district
    percentages = [race / total * 100 if total != 0 else 0 for race, total in zip(race_values, population_values)]

    return percentages

# SeaWulf-10. Identify opportunity districts in each random district plan (required) 
def count_opportunity_districts(percentages, threshold):
    # Count districts where the racial percentage is greater than the threshold
    return sum(1 for percent in percentages if percent >= threshold)

# # Example usage within the Markov Chain loop:

# Example usage within the Markov Chain loop:
# thresholds = [37, 50, 44]  # Include specific thresholds necessary (40 is the race-specific threshold)
thresholds = [37, 44, 50]

# Data structures to store frequency counts for opportunity districts
opportunity_districts = {
    'white': {thr: [] for thr in thresholds},
    'black': {thr: [] for thr in thresholds},
    'asian': {thr: [] for thr in thresholds},
    'hispanic': {thr: [] for thr in thresholds},
}

print(opportunity_districts)

# Initialize the dictionary
min_max_opportunity_districts = {
    threshold: {
        race: {"minimum": 200, "maximum": 0}
        for race in racial_groups
    }
    for threshold in thresholds
}

# SeaWulf-12. Calculate box & whisker data (required) (AD)

# Initialize a dictionary to store sorted percentages for each racial group
sorted_percentages = {race: [] for race in racial_groups}

# Start timing the Markov Chain execution
start_time_chain = time.time()

for i, partition in enumerate(recom_chain.with_progress_bar()): #can also use enumerate(tqdm.tqdm(recom_chain)): instead of the .with_progress_bar()
    # print(f"Finished step {i+1}/{len(recom_chain)}", end="\r")
    # assignment_list.append(partition.assignment)
    # results = calculate_election_splits(partition)
    
    splits_results = calculate_election_splits(partition)
    # print("Splits: ", splits_results)
    winners_results = calculate_district_winners_using_election_splits(splits_results)
    # print("Winners: ", winners_results)
    
    #Count the number of "D" in the winner_results
    number_democratic_winners = winners_results.count("D")
    number_republican_winners = winners_results.count("R")
    total_democratic_district_wins += number_democratic_winners
    total_republican_district_wins += number_republican_winners
    
    if(number_democratic_winners > maximum_democratic_winners):
        maximum_democratic_winners = number_democratic_winners
        print(f"New maximum Democratic winners: {number_democratic_winners}")
        minimum_republican_winners = number_republican_winners
        print(f"New minimum Republican winners: {number_republican_winners}")
        identify_and_store_district_plans(partition, f"./{state}interestingdistricts/votes/D{number_democratic_winners}_R{number_republican_winners}")
    
    if(number_democratic_winners < minimum_democratic_winners):
        minimum_democratic_winners = number_democratic_winners
        print(f"New minimum Democratic winners: {number_democratic_winners}")
        maximum_republican_winners = number_republican_winners
        print(f"New maximum Republican winners: {number_republican_winners}")
        identify_and_store_district_plans(partition, f"./{state}interestingdistricts/votes/D{number_democratic_winners}_R{number_republican_winners}")
    
    # print(partition["population"].values())

    # print(partition["white"])
    # print(partition["white"].values())
    # print(partition["population"].values())
    
    for race in racial_groups:
        race_percentages = calculate_racial_percentages(partition, race)
        sorted_race_percentages = sorted(race_percentages)
        sorted_percentages[race].append(sorted_race_percentages)
        
        # print(sorted_race_percentages)
        for thr in thresholds:
            num_opportunity_districts = count_opportunity_districts(sorted_race_percentages, thr)
            
            #flags for if we reach a new minimum or maximum for the race and threshold
            min_flag = False
            max_flag = False
            
            #If the number of opportunity districts for a particular race is less than the current minimum, update the minimum, and if it is greater than the current maximum, update the maximum. Also, call a function to save the district plan 
            if(num_opportunity_districts < min_max_opportunity_districts[thr][race]["minimum"]):
                # identify_and_store_district_plans(partition)
                min_max_opportunity_districts[thr][race]["minimum"] = num_opportunity_districts
                min_flag = True
            
            if(num_opportunity_districts > min_max_opportunity_districts[thr][race]["maximum"]):
                # identify_and_store_district_plans(partition)
                min_max_opportunity_districts[thr][race]["maximum"] = num_opportunity_districts
                max_flag = True
            
            # if(min_flag or max_flag):
                print(f"Reached a new minimum: {min_flag} or maximum: {max_flag} for race: {race} and threshold: {thr}. The number of opportunity districts is: {num_opportunity_districts}")
                identify_and_store_district_plans(partition, f"./{state}interestingdistricts/{race}/{race}_{thr}_{num_opportunity_districts}")
                # pass
            
            opportunity_districts[race][thr].append(num_opportunity_districts)
        continue
    # identify_and_store_district_plans(partition)
    # opp_districts = identify_opportunity_districts(partition, racial_key, thresholds)
    # print(f"Opportunity Districts for thresholds {thresholds}: {opp_districts}")
    continue

# After the loop, convert lists of lists into pandas DataFrames
# black_percentages_df = pd.DataFrame(black_percentages)
percentages_df = {race: pd.DataFrame(sorted_percentages[race]) for race in racial_groups}

# This will result in a DataFrame where each row corresponds to one partition,
# and columns are sorted percentages from the lowest to the highest among the districts in that partition.

# Calculate opportunity districts in the initial plan
initial_opportunity_districts = {
    race: {
        thr: count_opportunity_districts(calculate_racial_percentages(initial_partition, race), thr)
        for thr in thresholds
    } for race in racial_groups
}

print(initial_opportunity_districts)

# Calculate average opportunity districts across all generated plans
average_opportunity_districts = {
    race: {
        thr: sum(opportunity_districts[race][thr]) / len(opportunity_districts[race][thr])
        for thr in thresholds
    } for race in racial_groups
}
print("AVERAGEOPPORTUNITYDISTRICTS")
print(average_opportunity_districts)

# Stop timing after generating the plans
end_time_chain = time.time()

# Calculate and print the time taken to generate the plans
time_taken_chain = end_time_chain - start_time_chain
print(f"Time taken to generate {number_of_plans} plans: {time_taken_chain} seconds")

#Find the plan that had the most total opportunity districts for each threshold and save the partition cannot be done because the partition is not hashable outside of the loop
# If I could, I would save the partition to a file and then load it in the calculate_ensemble_measures function, but I cannot do that here. 
# #I actually mean that I would save the partition to a file in the identify_and_store_district_plans function, but we will have to compromise and not do that for now.

#Find the plan that had the most total opportunity districts for each threshold.
# max_opportunity_districts = {thr: 0 for thr in thresholds}
#Get the maximum and total opportunity districts for each threshold

print(opportunity_districts)

maximum_total_opportunity_districts = {thr: 0 for thr in thresholds}
minimum_total_opportunity_districts = {thr: 0 for thr in thresholds}
average_total_opportunity_districts = {thr: 0 for thr in thresholds}
# average_democratic_splits =
# average_republican_splits =

for thr in thresholds:
    totals_per_threshold = [
        sum(opportunity_districts[race][thr][i] for race in opportunity_districts)
        for i in range(number_of_plans)
    ]
    maximum_total_opportunity_districts[thr] = max(totals_per_threshold)
    minimum_total_opportunity_districts[thr] = min(totals_per_threshold)
    average_total_opportunity_districts[thr] = sum(totals_per_threshold) / len(totals_per_threshold)

average_democratic_district_wins = total_democratic_district_wins / number_of_plans
average_republican_district_wins = total_republican_district_wins / number_of_plans

# SeaWulf-7. Calculate ensemble measures (required) (AD)
# Calculate the summary measures for each ensemble. At a minimum, measures will include the
# number of district plans, the number of opportunity districts for each significant minority, and
# Republican/Democratic splits.
def calculate_ensemble_measures():
    #Save the number of district plans generated, the number of opportunity districts, and Republican/Democratic splits into a JSON file
    ensemble_results = {
        "number_of_plans": number_of_plans,
        "population_equality_threshold": population_epsilon,
        # "opportunity_districts_by_race_and_threshold": opportunity_districts,
        # "min_max_opportunity_districts": min_max_opportunity_districts,
        "maximum_democratic_winners": maximum_democratic_winners,
        "maximum_republican_winners": maximum_republican_winners,
        "minimum_democratic_winners": minimum_democratic_winners,
        "minimum_republican_winners": minimum_republican_winners,
        # "average_democratic_splits": average_democratic_splits,
        # "average_republican_splits": average_republican_splits,
        "average_democratic_district_wins": average_democratic_district_wins,
        "average_republican_district_wins": average_republican_district_wins,
        #Still need the number of opportunity districts for each significant minority and Republican/Democratic splits
        "maximum_total_opportunity_districts": maximum_total_opportunity_districts,
        "minimum_total_opportunity_districts": minimum_total_opportunity_districts,
        "average_total_opportunity_districts": average_total_opportunity_districts,
        # "average_democratic_splits": average_democratic_splits,
        # "average_republican_splits": average_republican_splits,
    }
    print(ensemble_results)
    
    # Save the ensemble results to a JSON file
    with open(f"./ensemblesummarydata/ensemble_results_{state}_{number_of_plans}_{node_repetitions}.json", "w") as file:
        json.dump(ensemble_results, file)
    # pass

calculate_ensemble_measures()

# Preparing data for comparison plot
data_for_plotting = []
for race in racial_groups:
    for thr in thresholds:
        data_for_plotting.append({
            'Race': race,
            'Threshold': thr,
            'Average Opportunity Districts': average_opportunity_districts[race][thr],
            'Initial Plan Opportunity Districts': initial_opportunity_districts[race][thr],
            'Type': 'Average Generated'
        })
        data_for_plotting.append({
            'Race': race,
            'Threshold': thr,
            'Average Opportunity Districts': initial_opportunity_districts[race][thr],
            'Initial Plan Opportunity Districts': initial_opportunity_districts[race][thr],
            'Type': 'Initial Plan'
        })

df_plot = pd.DataFrame(data_for_plotting)
print(df_plot)

def plot_opportunity_districts():
    # Assuming df_plot is already created
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        x='Race',
        y='Average Opportunity Districts',
        hue='Type',
        col='Threshold',  # Creates a separate subplot for each threshold
        data=df_plot,
        kind='bar',
        height=5,
        aspect=1
    )

    g.fig.subplots_adjust(top=0.9)  # adjust the Figure in sns.catplot()
    g.fig.suptitle(f'Comparison of Opportunity Districts by Race and Threshold for {state} with {number_of_plans} Plans')
    g.set_axis_labels("Race", "Number of Opportunity Districts")
    # g.set_xticklabels(rotation=45)
    # g.add_legend(title="Plan Type")

    plt.show()

plot_opportunity_districts()

def plot_box_whisker(data, title, ylabel, racial_key):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw 50% line for reference, if needed
    ax.axhline(0.5, color="#cccccc", linestyle="--")

    # Draw boxplot
    data.boxplot(ax=ax, positions=range(len(data.columns)))  # Corrected to get the number of columns directly

    # Assuming initial_partition is your initial state
    initial_percents = sorted(calculate_racial_percentages(initial_partition, racial_key))  # Again, adjust for 'NH_ASIAN' as needed
    plt.plot(initial_percents, "ro", label='Initial Plan')  # Red dots for the initial plan

    # Annotate and beautify the plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Sorted Districts")
    # ax.set_ylim(0, 100)  # Adjust based on the percentages scale you expect
    ax.legend()

    plt.xticks(rotation=45)
    plt.show()

# # Example plotting for each racial group
for race in racial_groups:
    plot_box_whisker(percentages_df[race], f'Distribution of {race.capitalize()} Population Across Redistricting Plans', f'% of {race.capitalize()} Population', race)
    continue