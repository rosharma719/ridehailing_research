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
