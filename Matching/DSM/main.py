import utils
import GurobiQB as qb
import eventgenerator as eg
import matchers
from offline_solver import solve_offline_optimal_with_markov

# Parameters
num_nodes = 5
simulation_time = 20

rate_riders = 0.4  
rate_drivers = 0.5
sojourn_rate_riders = 0.5
sojourn_rate_drivers = 0.2 
reward_value = num_nodes + 2
distance_penalty = 1

# Grid topology
grid_matrix = utils.generate_grid_adjacency_matrix(num_nodes)
print("GRID\n", grid_matrix)

# Ring topology
ring_matrix = utils.generate_ring_adjacency_matrix(num_nodes)
print("RING\n", ring_matrix)

# Compute rewards separately for each topology
grid_rewards = utils.divided_adjacency_to_rewards(grid_matrix, reward_value, denom_multiple=1)
ring_rewards = utils.divided_adjacency_to_rewards(ring_matrix, reward_value, denom_multiple=1)

# Define arrival rates for active and passive types based on nodes
lambda_i = {f"Active Driver Node {i}": rate_drivers for i in range(num_nodes)}
lambda_j = {f"Passive Rider Node {i}": rate_riders for i in range(num_nodes)}

# Define abandonment rates for active types (example values)
mu_i = {f"Active Driver Node {i}": sojourn_rate_drivers for i in range(num_nodes)}

# Obtain separate QB optimization results for each topology
print("\n\nGRID RESULTS\n\n")
QB_results_grid = qb.solve_QB(grid_rewards, lambda_i, lambda_j, mu_i)
print("\n\nRING RESULTS\n\n")
QB_results_ring = qb.solve_QB(ring_rewards, lambda_i, lambda_j, mu_i)

def run_stuff(event_queue, adjacency_matrix, rewards, QB_results, topology_name):
    print(f"\n\n\n\n{topology_name.upper()}\n\n")
    
    # Clone the event queue for multiple simulations
    event_queue_2 = event_queue.clone()
    event_queue_3 = event_queue.clone()
    event_queue_4 = event_queue.clone()
    event_queue_5 = event_queue.clone()

    print("GREEDY AUTO-LABEL\n")
    matchers.greedy_auto_label(event_queue_2, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("GREEDY SOLUTION\n")
    matchers.greedy_matcher(event_queue_3, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("\nOFFLINE OPTIMAL SOLUTION\n")
    solve_offline_optimal_with_markov(event_queue_4, rewards)

# Generate event queue
event_queue = eg.EventQueue()
eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

# Run for ring topology with separate QB results
#run_stuff(event_queue, ring_matrix, ring_rewards, QB_results_ring, "ring")

# Run for grid topology with separate QB results
run_stuff(event_queue, grid_matrix, grid_rewards, QB_results_grid, "grid")
