import utils
import GurobiQB as qb
import eventgenerator as eg
import matchers
from offline_solver import solve_offline_optimal_with_markov
from logchecker import *
import logging

# ======= CONFIGURE LOGGING (OVERWRITES PREVIOUS LOGS) =======
logging.basicConfig(
    filename="logs.txt",  
    filemode="w",  # Overwrites the file instead of appending
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG   
)


# ======= USER-SPECIFIED TOPOLOGY MODE =======
topology_mode = "grid"  # Choose from: "grid", "ring", "both"
reward_structure = "increasing_distance_penalty"  # Choose from: "divided", "standard", "increasing_distance_penalty"
'''
Standard reward: Base reward - distance_penalty
Divided reward: Base reward/(1+denominator), set in grid_denom or ring_denom
Increasing distance penalty: Base reward - 2*distance
'''


# ======= PARAMETERS =======
num_nodes = 10
simulation_time = 20

rate_riders = 0.4  
rate_drivers = 0.5
sojourn_rate_riders = 0.5
sojourn_rate_drivers = 0.2 
reward_value = num_nodes*0.75
distance_penalty = 0.5

# Separate denominator parameters for different topologies
grid_denom = 1  # Grid topology denominator multiplier
ring_denom = 1  # Ring topology denominator multiplier

# Thickness floor for Threshold Nonperish GAL
thickness_floor = 1

# ======= GENERATE ADJACENCY MATRICES =======
grid_matrix = utils.generate_grid_adjacency_matrix(num_nodes)
ring_matrix = utils.generate_ring_adjacency_matrix(num_nodes)

print("\nGRID TOPOLOGY\n", grid_matrix)
#print("\nRING TOPOLOGY\n", ring_matrix)

# ======= CHOOSE REWARD STRUCTURE BASED ON SWITCHER =======
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
lambda_i = {f"Active Driver Node {i}": rate_drivers for i in range(num_nodes)}
lambda_j = {f"Passive Rider Node {i}": rate_riders for i in range(num_nodes)}
mu_i = {f"Active Driver Node {i}": sojourn_rate_drivers for i in range(num_nodes)}

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
    
    # Clone the event queue for multiple simulations
    event_queue_2 = event_queue.clone()
    event_queue_3 = event_queue.clone()
    event_queue_4 = event_queue.clone()
    event_queue_5 = event_queue.clone()
    event_queue_6 = event_queue.clone()

    print("\nGREEDY AUTO-LABEL\n")
    matchers.greedy_auto_label(event_queue_2, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("\nGREEDY SOLUTION\n")
    matchers.greedy_matcher(event_queue_3, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("\nGREEDY AUTO-LABEL NONPERISHING\n")
    matchers.nonperish_bidirectional_GAL(event_queue_4, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    '''print("\nGREEDY AUTO-LABEL NONPERISHING WITH THRESHOLD\n")
    matchers.threshold_bidirectional_GAL(event_queue_5, rewards, QB_results, lambda_i, lambda_j, mu_i, thickness_floor, adjacency_matrix)
    '''
    
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

