import numpy as np
import networkx as nx
import random
import pandas as pd
from scipy.optimize import linear_sum_assignment


# CITY LAYOUT AND GRAPH FUNCTIONS

def generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob=0.15, extra_edges=0.15):
    if num_nodes <= 1:
        raise ValueError("Number of nodes must be greater than 1")

    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        if i % grid_size != grid_size - 1 and i + 1 < num_nodes:
            if random.random() > skip_prob:
                adjacency_matrix[i][i + 1] = 1
                adjacency_matrix[i + 1][i] = 1
        if i + grid_size < num_nodes:
            if random.random() > skip_prob:
                adjacency_matrix[i][i + grid_size] = 1
                adjacency_matrix[i + grid_size][i] = 1

    num_extra_edges = int(extra_edges * num_nodes)
    edges_added = 0
    while edges_added < num_extra_edges:
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node1 != node2 and adjacency_matrix[node1][node2] == 0:
            adjacency_matrix[node1][node2] = 1
            adjacency_matrix[node2][node1] = 1
            edges_added += 1

    return adjacency_matrix

# Poisson Arrivals and Exponential Sojourn Times

def generate_poisson_arrivals(rate, shape):
    return np.random.poisson(rate, shape)

def generate_exponential_sojourn_times(rate, length):
    return np.random.exponential(1 / rate, length)

# Rider and Driver Classes

class Rider:
    def __init__(self, location, patience, sojourn_time):
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time

    def update_patience(self):
        if self.patience > 0:
            self.patience -= 1

class Driver:
    def __init__(self, location, patience, sojourn_time):
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time

    def update_patience(self):
        if self.patience > 0:
            self.patience -= 1

# Function to create Riders and Drivers from series data

def create_riders_and_drivers(pickup_series, dropoffs, drivers_series, sojourn_times):
    riders = {}
    drivers = {}
    for t in range(len(pickup_series)):
        riders[t] = []
        drivers[t] = []
        for node in range(len(pickup_series[t])):
            for _ in range(pickup_series[t, node]):
                dropoff = random.choice(dropoffs.get((t, node), []))
                patience = random.randint(1, 10)
                sojourn_time = sojourn_times[t]
                riders[t].append(Rider(location=(t, node), patience=patience, sojourn_time=sojourn_time))
            for _ in range(drivers_series[t, node]):
                patience = random.randint(1, 10)
                sojourn_time = sojourn_times[t]
                drivers[t].append(Driver(location=(t, node), patience=patience, sojourn_time=sojourn_time))
    return riders, drivers

# Function to store object status in a dataframe

def create_status_dataframe(riders, drivers):
    data = []
    for t in riders.keys():
        for rider in riders[t]:
            data.append({
                'Time': t,
                'Type': 'Rider',
                'Location': rider.location,
                'Patience': rider.patience,
                'Sojourn Time': rider.sojourn_time
            })
        for driver in drivers[t]:
            data.append({
                'Time': t,
                'Type': 'Driver',
                'Location': driver.location,
                'Patience': driver.patience,
                'Sojourn Time': driver.sojourn_time
            })
    return pd.DataFrame(data)


def optimal_offline_benchmark(riders, drivers, adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))

    all_riders = []
    all_drivers = []

    # Collect all riders and drivers
    for t, rider_list in riders.items():
        all_riders.extend(rider_list)
    for t, driver_list in drivers.items():
        all_drivers.extend(driver_list)

    # Create cost matrix
    cost_matrix = np.full((len(all_riders), len(all_drivers)), np.inf)

    for i, rider in enumerate(all_riders):
        for j, driver in enumerate(all_drivers):
            rider_pos = rider.location[1]
            driver_pos = driver.location[1]
            travel_cost = path_lengths[rider_pos][driver_pos]
            cost_matrix[i, j] = travel_cost

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_riders = set()
    matched_drivers = set()

    for r, d in zip(row_ind, col_ind):
        if cost_matrix[r, d] < np.inf:  # Valid match
            matched_riders.add(all_riders[r])
            matched_drivers.add(all_drivers[d])

    unmatched_riders = set(all_riders) - matched_riders
    unmatched_drivers = set(all_drivers) - matched_drivers

    # Calculate total wait time for matched riders and drivers
    total_wait_time_riders = sum(rider.sojourn_time for rider in matched_riders)
    total_wait_time_drivers = sum(driver.sojourn_time for driver in matched_drivers)

    # Summary statistics
    total_riders = len(all_riders)
    total_drivers = len(all_drivers)
    matched_riders_count = len(matched_riders)
    unmatched_riders_count = len(unmatched_riders)
    matched_drivers_count = len(matched_drivers)
    unmatched_drivers_count = len(unmatched_drivers)
    average_wait_time_riders = total_wait_time_riders / matched_riders_count if matched_riders_count else 0
    average_wait_time_drivers = total_wait_time_drivers / matched_drivers_count if matched_drivers_count else 0

    # Print results
    print("Summary Statistics:")
    print(f"Total Riders: {total_riders}")
    print(f"Matched Riders: {matched_riders_count}")
    print(f"Unmatched Riders: {unmatched_riders_count}")
    print(f"Total Drivers: {total_drivers}")
    print(f"Matched Drivers: {matched_drivers_count}")
    print(f"Unmatched Drivers: {unmatched_drivers_count}")
    print(f"Average Wait Time for Riders: {average_wait_time_riders:.2f}")
    print(f"Average Wait Time for Drivers: {average_wait_time_drivers:.2f}")

    return matched_riders, unmatched_riders, matched_drivers, unmatched_drivers


# Just adjust arrival parameter rather than adding a price mechanism into the matching model
# Make the arrival rates seasonal and non-static 

# Recalculate LP benchmark with every parameter change
# "Upon statistical difference, recalculate"
# Arrival rate is a parameter of the 