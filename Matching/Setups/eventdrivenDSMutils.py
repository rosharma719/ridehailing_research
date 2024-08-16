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

# Event-Driven System

class Event:
    def __init__(self, event_time, event_type, obj):
        self.event_time = event_time
        self.event_type = event_type
        self.obj = obj

class EventQueue:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)
        self.events.sort(key=lambda x: x.event_time)

    def get_next_event(self):
        return self.events.pop(0) if self.events else None

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

# Function to process events

def process_event(event_queue, riders, drivers, adj_matrix, batch_window):
    while event_queue.events:
        event = event_queue.get_next_event()
        if event.event_type == "rider_arrival":
            # Perform matching or add to waiting list
            pass
        elif event.event_type == "driver_arrival":
            # Perform matching or add to waiting list
            pass
        elif event.event_type == "batch_match":
            matched_riders, unmatched_riders, matched_drivers, unmatched_drivers = batch_matching(riders, drivers, adj_matrix, batch_window)
            # Update system with matched and unmatched riders/drivers

def greedy_matching(riders, drivers):
    matched_riders = set()
    matched_drivers = set()

    # Perform greedy matching
    for t in riders.keys():
        for rider in riders[t]:
            for driver in drivers[t]:
                if driver not in matched_drivers and driver.location == rider.location and driver.patience > 0:
                    matched_riders.add(rider)
                    matched_drivers.add(driver)
                    break

    # Calculate unmatched riders and drivers
    all_riders = set(r for t_list in riders.values() for r in t_list)
    all_drivers = set(d for t_list in drivers.values() for d in t_list)
    unmatched_riders = all_riders - matched_riders
    unmatched_drivers = all_drivers - matched_drivers

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

def batch_matching(riders, drivers, adj_matrix, batch_window):
    G = nx.from_numpy_array(adj_matrix)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))

    matched_riders = set()
    matched_drivers = set()

    all_riders = []
    all_drivers = []

    # Process in batches based on the batch window
    for t in range(0, len(riders), batch_window):
        batch_riders = []
        batch_drivers = []

        # Collect all riders and drivers in the current batch window
        for batch_t in range(t, min(t + batch_window, len(riders))):
            batch_riders.extend(riders[batch_t])
            batch_drivers.extend(drivers[batch_t])

        # Find optimal matching within the current batch window
        for rider in batch_riders:
            best_match = None
            best_cost = float('inf')

            for driver in batch_drivers:
                if driver not in matched_drivers:
                    rider_pos = rider.location[1]
                    driver_pos = driver.location[1]
                    travel_cost = path_lengths[rider_pos][driver_pos]

                    if travel_cost < best_cost:
                        best_cost = travel_cost
                        best_match = driver

            if best_match:
                matched_riders.add(rider)
                matched_drivers.add(best_match)
                batch_drivers.remove(best_match)  # Remove matched driver from the batch

        all_riders.extend(batch_riders)
        all_drivers.extend(batch_drivers)

    # Calculate unmatched riders and drivers
    unmatched_riders = set(all_riders) - matched_riders
    unmatched_drivers = set(all_drivers) - matched_drivers

    # Calculate total wait time for matched
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
    print("Batch Matching Summary Statistics:")
    print(f"Total Riders: {total_riders}")
    print(f"Matched Riders: {matched_riders_count}")
    print(f"Unmatched Riders: {unmatched_riders_count}")
    print(f"Total Drivers: {total_drivers}")
    print(f"Matched Drivers: {matched_drivers_count}")
    print(f"Unmatched Drivers: {unmatched_drivers_count}")
    print(f"Average Wait Time for Riders: {average_wait_time_riders:.2f}")
    print(f"Average Wait Time for Drivers: {average_wait_time_drivers:.2f}")

    return matched_riders, unmatched_riders, matched_drivers, unmatched_drivers
