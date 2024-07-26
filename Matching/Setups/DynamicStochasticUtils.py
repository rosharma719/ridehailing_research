import numpy as np
import networkx as nx
import random
import pandas as pd

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
