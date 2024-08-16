from Setups.eventdrivenDSMutils import *

# Parameters
num_nodes = 10
skip_prob = 0.15
extra_edges = 0.15
rate_riders = 5  # Rate of rider arrivals (Poisson)
rate_drivers = 5  # Rate of driver arrivals (Poisson)
sojourn_rate = 0.4
batch_window = 5  # Batch window size
simulation_time = 100  # Total simulation time

# Generate city graph
adj_matrix = generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)

# Initialize event queue
event_queue = EventQueue()

# Generate events
generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate, num_nodes, simulation_time)

# Perform greedy matching
print("\nGREEDY MATCHING")
matched_pairs_greedy, unmatched_riders_greedy, unmatched_drivers_greedy = greedy_matching_process(event_queue, adj_matrix)
greedy_results = analyze_results(matched_pairs_greedy, unmatched_riders_greedy, unmatched_drivers_greedy)

# Reset event queue for batch matching
event_queue = EventQueue()
generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate, num_nodes, simulation_time)

# Perform batch matching
print("\nBATCH MATCHING")
matched_pairs_batch, unmatched_riders_batch, unmatched_drivers_batch = batch_matching_process(event_queue, adj_matrix, batch_window)
batch_results = analyze_results(matched_pairs_batch, unmatched_riders_batch, unmatched_drivers_batch)

# Compare results
print("\nCOMPARISON")
print(f"Greedy Matching - Matched: {greedy_results['matched_count']}, Avg Wait Time: {greedy_results['avg_wait_time']:.2f}")
print(f"Batch Matching - Matched: {batch_results['matched_count']}, Avg Wait Time: {batch_results['avg_wait_time']:.2f}")