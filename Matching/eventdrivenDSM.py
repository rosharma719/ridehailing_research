from Setups.eventdrivenDSMutils import *

# Parameters
num_nodes = 10
skip_prob = 0.15
extra_edges = 0.15
rate_riders = 5  # Rate of rider arrivals (Poisson)
rate_drivers = 5  # Rate of driver arrivals (Poisson)
sojourn_rate = 0.4
batch_window = 5  # Batch window size

# Initialize event queue and other data structures
event_queue = []
matched_riders = []
matched_drivers = []

# Generate city graph
adj_matrix = generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)

# Generate riders and drivers as events
create_riders_and_drivers(event_queue, rate_riders, rate_drivers, sojourn_rate, adj_matrix, num_nodes)

# Process events and perform matching
matched_riders, matched_drivers = process_event(event_queue, matched_riders, matched_drivers, adj_matrix)

# Perform batch matching
print("\n\nBATCH MATCHING \n")
matched_riders_batch, unmatched_riders_batch, matched_drivers_batch, unmatched_drivers_batch = batch_matching(matched_riders, matched_drivers, adj_matrix, batch_window)

# Perform greedy matching
print("\n\nGREEDY MATCHING \n")
matched_riders_greedy, unmatched_riders_greedy, matched_drivers_greedy, unmatched_drivers_greedy = greedy_matching(matched_riders, matched_drivers)

