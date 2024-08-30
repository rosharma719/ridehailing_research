import utils
import GurobiRBandQB as grb
import eventgenerator as eg
import matchers

# Parameters
num_nodes = 10
skip_prob = 0 # Probability of skipping an edge in the grid
extra_edges = 0 # Proportion of extra random edges in the graph

# Generate the adjacency matrix and reward matrix
adj_matrix = utils.generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)
rewards = utils.adjacency_to_rewards(adj_matrix, reward_value=8)


# Define arrival rates for active and passive types based on nodes
lambda_i = {f"Active Driver Node {i}": 1.0 for i in range(num_nodes)}
lambda_j = {f"Passive Rider Node {i}": 0.8 for i in range(num_nodes)}

# Solve RB to get the flow matrix
flow_matrix = grb.solve_RB(rewards, lambda_i, lambda_j)

# Calculate label distribution
active_types = [f"Active Driver Node {i}" for i in range(num_nodes)]
passive_types = [f"Passive Rider Node {i}" for i in range(num_nodes)]


# Event Generation Parameters
simulation_time = 100  

rate_riders = 0.3  # Arrival rate of riders
rate_drivers = 0.4  # Arrival rate of drivers

sojourn_rate_riders = 0.5  # Average sojourn time for riders
sojourn_rate_drivers = 0.2  # Average sojourn time for drivers

# Generate the event timeline using the label distribution
event_queue = eg.EventQueue()
eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

def print_events(event_queue):
    # Display generated events in order
    print("\nGenerated Events:")
    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        entity_type = event.entity.type if event.entity else 'Unknown'
        print(f"Time: {event.time:.2f}, Event: {event.event_type}, Entity: {entity_type}, Location: {event.entity.location}")

# Call the greedy matching algorithm
matchers.greedy_auto_label(event_queue)