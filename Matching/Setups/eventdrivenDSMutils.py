import numpy as np
import networkx as nx
import random
import pandas as pd
import heapq
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

# Rider and Driver Classes

class Rider:
    def __init__(self, arrival_time, location, patience, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time

class Driver:
    def __init__(self, arrival_time, location, patience, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time

# Event-Driven System

class Event:
    def __init__(self, time, event_type, entity=None):
        self.time = time
        self.event_type = event_type
        self.entity = entity

    def __lt__(self, other):
        return self.time < other.time

class EventQueue:
    def __init__(self):
        self.queue = []

    def add_event(self, event):
        heapq.heappush(self.queue, event)

    def get_next_event(self):
        return heapq.heappop(self.queue) if self.queue else None

# Event Generation

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate, num_nodes, simulation_time):
    current_time = 0
    while current_time < simulation_time:
        # Generate rider arrivals
        next_rider_time = current_time + np.random.exponential(1/rate_riders)
        if next_rider_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate)
            rider_location = np.random.randint(0, num_nodes)
            rider = Rider(next_rider_time, rider_location, patience, sojourn_time)
            event_queue.add_event(Event(next_rider_time, 'rider_arrival', rider))

        # Generate driver arrivals
        next_driver_time = current_time + np.random.exponential(1/rate_drivers)
        if next_driver_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate)
            driver_location = np.random.randint(0, num_nodes)
            driver = Driver(next_driver_time, driver_location, patience, sojourn_time)
            event_queue.add_event(Event(next_driver_time, 'driver_arrival', driver))

        current_time = min(next_rider_time, next_driver_time)

# Matching Functions

def greedy_matching_process(event_queue, adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    
    available_riders = []
    available_drivers = []
    matched_pairs = []
    current_time = 0

    while True:
        event = event_queue.get_next_event()
        if event is None:
            break

        current_time = event.time

        if event.event_type == 'rider_arrival':
            rider = event.entity
            match = find_best_match(rider, available_drivers, path_lengths)
            if match:
                driver, _ = match
                matched_pairs.append((rider, driver, current_time))
                available_drivers.remove(driver)
            else:
                available_riders.append(rider)
        elif event.event_type == 'driver_arrival':
            driver = event.entity
            match = find_best_match(driver, available_riders, path_lengths)
            if match:
                rider, _ = match
                matched_pairs.append((rider, driver, current_time))
                available_riders.remove(rider)
            else:
                available_drivers.append(driver)

        # Remove expired riders and drivers
        available_riders = [r for r in available_riders if current_time - r.arrival_time < r.patience]
        available_drivers = [d for d in available_drivers if current_time - d.arrival_time < d.patience]

    return matched_pairs, available_riders, available_drivers

def find_best_match(entity, available_entities, path_lengths):
    best_match = None
    best_cost = float('inf')

    for other_entity in available_entities:
        cost = path_lengths[entity.location][other_entity.location]
        if cost < best_cost:
            best_cost = cost
            best_match = other_entity

    return (best_match, best_cost) if best_match else None

def batch_matching_process(event_queue, adj_matrix, batch_window):
    G = nx.from_numpy_array(adj_matrix)
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    
    batch_riders = []
    batch_drivers = []
    matched_pairs = []
    current_time = 0
    next_batch_time = batch_window

    while True:
        event = event_queue.get_next_event()
        if event is None:
            break

        current_time = event.time

        if event.event_type == 'rider_arrival':
            batch_riders.append(event.entity)
        elif event.event_type == 'driver_arrival':
            batch_drivers.append(event.entity)

        if current_time >= next_batch_time:
            new_matches = perform_batch_matching(batch_riders, batch_drivers, path_lengths, current_time)
            matched_pairs.extend(new_matches)
            
            # Remove matched entities from batches
            batch_riders = [r for r in batch_riders if not any(m[0] == r for m in new_matches)]
            batch_drivers = [d for d in batch_drivers if not any(m[1] == d for m in new_matches)]
            
            next_batch_time = current_time + batch_window

    return matched_pairs, batch_riders, batch_drivers

def perform_batch_matching(riders, drivers, path_lengths, current_time):
    cost_matrix = np.zeros((len(riders), len(drivers)))
    for i, rider in enumerate(riders):
        for j, driver in enumerate(drivers):
            cost_matrix[i][j] = path_lengths[rider.location][driver.location]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        matches.append((riders[r], drivers[c], current_time))

    return matches

# Analysis Function

def analyze_results(matched_pairs, unmatched_riders, unmatched_drivers):
    total_riders = len(matched_pairs) + len(unmatched_riders)
    total_drivers = len(matched_pairs) + len(unmatched_drivers)
    
    matched_count = len(matched_pairs)
    unmatched_riders_count = len(unmatched_riders)
    unmatched_drivers_count = len(unmatched_drivers)
    
    total_wait_time = sum(match_time - rider.arrival_time for rider, _, match_time in matched_pairs)
    avg_wait_time = total_wait_time / matched_count if matched_count > 0 else 0
    
    print(f"Total Riders: {total_riders}")
    print(f"Total Drivers: {total_drivers}")
    print(f"Matched Pairs: {matched_count}")
    print(f"Unmatched Riders: {unmatched_riders_count}")
    print(f"Unmatched Drivers: {unmatched_drivers_count}")
    print(f"Average Wait Time: {avg_wait_time:.2f}")

    return {
        'total_riders': total_riders,
        'total_drivers': total_drivers,
        'matched_count': matched_count,
        'unmatched_riders': unmatched_riders_count,
        'unmatched_drivers': unmatched_drivers_count,
        'avg_wait_time': avg_wait_time
    }