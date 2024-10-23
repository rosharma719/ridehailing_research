import utils
import GurobiRBandQB as grb
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
reward_value = num_nodes + 1

# Generate the adjacency matrix and reward matrix
adj_matrix = utils.generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)
rewards = utils.adjacency_to_rewards(adj_matrix, reward_value)

# Define arrival rates for active and passive types based on nodes
lambda_i = {f"Active Driver Node {i}": rate_drivers for i in range(num_nodes)}
lambda_j = {f"Passive Rider Node {i}": rate_riders for i in range(num_nodes)}

# Define abandonment rates for active types (example values)
mu_i = {f"Active Driver Node {i}": sojourn_rate_drivers for i in range(num_nodes)}

# Obtain flow matrix from QB optimization
QB_flow_matrix = grb.solve_QB(rewards, lambda_i, lambda_j, mu_i)['flow_matrix']

# Function to run the simulation
def run_stuff():
    event_queue = eg.EventQueue()
    eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)
    
    # Call the greedy_auto_label with the flow matrix from the QB optimization
    matchers.greedy_auto_label(event_queue, rewards, QB_flow_matrix)

run_stuff()