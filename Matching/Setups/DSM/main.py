import utils
import GurobiRBandQB as grb
import eventgenerator as eg

# Parameters


# Example usage
num_nodes = 10
adj_matrix = utils.generate_imperfect_grid_adjacency_matrix(num_nodes)
rewards = utils.adjacency_to_rewards(adj_matrix)

# Define arrival rates for active and passive types based on nodes
lambda_i = {f"Active Driver {i}": 1.0 for i in range(num_nodes)}
lambda_j = {f"Passive Rider {i}": 0.8 for i in range(num_nodes)}

grb.solve_RB(rewards, lambda_i, lambda_j)

# Example Usage
simulation_time = 100  # Simulation end time
num_nodes = 10  # Number of locations in the city grid
rate_riders = 0.5  # Arrival rate of riders
rate_drivers = 0.5  # Arrival rate of drivers
sojourn_rate_riders = 0.1  # Average sojourn time for riders
sojourn_rate_drivers = 0.1  # Average sojourn time for drivers

event_queue = eg.EventQueue()
eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

# Display Generated Events in Order
while not event_queue.is_empty():
    event = event_queue.get_next_event()
    entity_type = event.entity.type if event.entity else 'Unknown'
    print(f"Time: {event.time:.2f}, Event: {event.event_type}, Entity: {entity_type}, Location: {event.entity.location}")

