import utils
import GurobiQB as qb
import eventgenerator as eg
import matchers

# Parameters
num_nodes = 2
skip_prob = 0  
extra_edges = 0  
simulation_time = 100  

rate_riders = 0.4  
rate_drivers = 0.3  
sojourn_rate_riders = 0.5  
sojourn_rate_drivers = 0.2  
reward_value = num_nodes + 2
distance_penalty = 1

# Generate the adjacency matrix and reward matrix
adj_matrix = utils.generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)
rewards = utils.adjacency_to_rewards(adj_matrix, reward_value, distance_penalty)

# Define arrival rates for active and passive types based on nodes
lambda_i = {f"Active Driver Node {i}": rate_drivers for i in range(num_nodes)}
lambda_j = {f"Passive Rider Node {i}": rate_riders for i in range(num_nodes)}

# Define abandonment rates for active types (example values)
mu_i = {f"Active Driver Node {i}": sojourn_rate_drivers for i in range(num_nodes)}

# Obtain full results from QB optimization
QB_results = qb.solve_QB(rewards, lambda_i, lambda_j, mu_i)

def run_stuff():
    event_queue = eg.EventQueue()
    eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

    # Clone the event queue for multiple simulations
    event_queue_2 = event_queue.clone()
    event_queue_3 = event_queue.clone()

    # Call the greedy_auto_label algorithms
    matchers.greedy_auto_label(event_queue, rewards, QB_results, lambda_i, lambda_j, mu_i, adj_matrix)
    matchers.greedy_auto_label_nonperish(event_queue_2, rewards, QB_results, lambda_i, lambda_j, mu_i, adj_matrix)
    matchers.greedy_auto_label_nonperish_floor(event_queue_3, rewards, QB_results, lambda_i, lambda_j, mu_i, thickness_floor=1, adjacency_matrix=adj_matrix)


run_stuff()
