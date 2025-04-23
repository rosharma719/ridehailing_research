import json
import utils
import GurobiQB as qb
import eventgenerator as eg
import numpy as np
import matchers
from offline_solver import solve_offline_optimal_with_markov
from logchecker import *
import logging


# ======= LOAD CONFIG =======
import os
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)



# ======= CONFIGURE LOGGING (OVERWRITES PREVIOUS LOGS) =======
logging.basicConfig(
    filename="logs.txt",  
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG   
)

# ======= LOAD PARAMETERS FROM CONFIG =======
topology_mode = config["topology_mode"]
reward_structure = config["reward_structure"]
num_nodes = config["num_nodes"]
simulation_time = config["simulation_time"]
rate_riders = config["rate_riders"]
rate_drivers = config["rate_drivers"]
sojourn_rate_riders = config["sojourn_rate_riders"]
sojourn_rate_drivers = config["sojourn_rate_drivers"]
distance_penalty = config["distance_penalty"]
grid_denom = config["grid_denom"]
ring_denom = config["ring_denom"]
thickness_floor = config["thickness_floor"]

reward_value = num_nodes + 1

# ======= GENERATE OR LOAD ADJACENCY AND REWARD MATRICES =======
use_custom = config.get("use_custom_matrices", False)

if use_custom:
    adj_path = os.path.join(os.path.dirname(__file__), config["custom_adjacency_path"])
    reward_path = os.path.join(os.path.dirname(__file__), config["custom_reward_path"])

    print(f"\nUSING CUSTOM MATRICES:\n  Adjacency: {adj_path}\n  Reward: {reward_path}")

    custom_adjacency = np.load(adj_path)
    with open(reward_path, "r") as f:
        custom_rewards = json.load(f)

    grid_matrix = ring_matrix = custom_adjacency
    grid_rewards = ring_rewards = custom_rewards
else:
    grid_matrix = utils.generate_grid_adjacency_matrix(num_nodes)
    ring_matrix = utils.generate_ring_adjacency_matrix(num_nodes)

    print("\nGRID TOPOLOGY\n", grid_matrix)

    if reward_structure == "divided":
        grid_rewards = utils.divided_adjacency_to_rewards(grid_matrix, reward_value, denom_multiple=grid_denom)
        ring_rewards = utils.divided_adjacency_to_rewards(ring_matrix, reward_value, denom_multiple=ring_denom)
    elif reward_structure == "standard":
        grid_rewards = utils.adjacency_to_rewards(grid_matrix, reward_value, distance_penalty)
        ring_rewards = utils.adjacency_to_rewards(ring_matrix, reward_value, distance_penalty)
    elif reward_structure == "increasing_distance_penalty":
        grid_rewards = utils.increasing_distance_penalty(grid_matrix, reward_value)
        ring_rewards = utils.increasing_distance_penalty(ring_matrix, reward_value)
    else:
        raise ValueError("Invalid reward structure. Choose from: 'divided', 'standard', 'increasing_distance_penalty'")

# ======= ARRIVAL & ABANDONMENT RATES =======
driver_node_names = [f"Active Driver Node {i}" for i in range(num_nodes)]
rider_node_names = [f"Passive Rider Node {i}" for i in range(num_nodes)]

use_custom_rates = config.get("use_custom_rates", False)
custom_rate_path = os.path.join(os.path.dirname(__file__), config.get("custom_rate_path", ""))

if use_custom_rates:
    print("USING CUSTOM RATES\n\n")
    with open(custom_rate_path, "r") as f:
        custom_rates = json.load(f)

    lambda_i = {name: custom_rates[name]["arrival"] for name in driver_node_names}
    lambda_j = {name: custom_rates[name]["arrival"] for name in rider_node_names}
    mu_i =     {name: custom_rates[name]["sojourn"] for name in driver_node_names}
else:
    print("USING DEFAULT RATES\n\n")
    lambda_i = utils.generate_param_normal_distribution(driver_node_names, mean=rate_drivers, std_dev=0.2, min_val=0.05, max_val=2, seed=42)
    lambda_j = utils.generate_param_normal_distribution(rider_node_names, mean=rate_riders, std_dev=0.2, min_val=0.05, max_val=2, seed=101)
    mu_i =     utils.generate_param_normal_distribution(driver_node_names, mean=1, std_dev=0.5, min_val=0.05, max_val=4, seed=7)

# ======= RUN QB OPTIMIZATION BASED ON TOPOLOGY MODE =======
QB_results_grid = None
QB_results_ring = None

if topology_mode in ["grid", "both"]:
    print("\n\nGRID RESULTS\n\n")
    QB_results_grid = qb.solve_QB(grid_rewards, lambda_i, lambda_j, mu_i)

if topology_mode in ["ring", "both"]:
    print("\n\nRING RESULTS\n\n")
    QB_results_ring = qb.solve_QB(ring_rewards, lambda_i, lambda_j, mu_i)

# ======= FUNCTION TO RUN MATCHING & OPTIMIZATION =======
def run_stuff(event_queue, adjacency_matrix, rewards, QB_results, topology_name):
    print(f"\n\n\n\n{topology_name.upper()}\n\n")
    
    event_queue_2 = event_queue.clone()
    event_queue_3 = event_queue.clone()
    event_queue_4 = event_queue.clone()
    event_queue_5 = event_queue.clone()
    event_queue_6 = event_queue.clone()

    print("\nGREEDY AUTO-LABEL\n")
    explicitly_remove_riders = True
    matchers.greedy_auto_label(event_queue_2, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix, explicitly_remove_riders)

    print("\nGREEDY SOLUTION\n")
    matchers.greedy_matcher(event_queue_3, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("\nGREEDY AUTO-LABEL NONPERISHING\n")
    matchers.nonperish_bidirectional_GAL(event_queue_4, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    # Uncomment if threshold logic is desired
    # print("\nGREEDY AUTO-LABEL NONPERISHING WITH THRESHOLD\n")
    # matchers.threshold_bidirectional_GAL(event_queue_5, rewards, QB_results, lambda_i, lambda_j, mu_i, thickness_floor, adjacency_matrix)

    print("\nOFFLINE OPTIMAL SOLUTION\n")
    solve_offline_optimal_with_markov(event_queue_6, rewards)

# ======= GENERATE EVENT QUEUE (SHARED) =======
event_queue = eg.EventQueue()
eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

num_arrivals, num_abandonments = utils.count_events(event_queue)
print(f"Total Arrivals: {num_arrivals}, Total Abandonments: {num_abandonments}")

# ======= RUN SIMULATIONS BASED ON TOPOLOGY MODE =======
if topology_mode in ["grid", "both"]:
    run_stuff(event_queue, grid_matrix, grid_rewards, QB_results_grid, "grid")

if topology_mode in ["ring", "both"]:
    run_stuff(event_queue, ring_matrix, ring_rewards, QB_results_ring, "ring")

print(parse_log_file('logs.txt'))
