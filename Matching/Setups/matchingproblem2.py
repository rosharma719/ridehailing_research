import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import seaborn as sns
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

def visualize_graph(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    pos = {i: (i % int(np.ceil(np.sqrt(len(adj_matrix)))), int(np.ceil(np.sqrt(len(adj_matrix)))) - (i // int(np.ceil(np.sqrt(len(adj_matrix)))))) for i in range(len(adj_matrix))}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title("Imperfect Grid-Like City Graph")
    plt.show()

# TRAFFIC AND TIME SERIES SIMULATION

def generate_seasonal_demand(length, num_nodes, mean, std_dev, amplitude_range, period_range, num_periods_range):
    seasonal_patterns = [generate_complex_seasonal_pattern(length, amplitude_range, period_range, num_periods_range) for _ in range(num_nodes)]
    demand = np.zeros((length, num_nodes), dtype=int)

    for t in range(length):
        for i in range(num_nodes):
            base_demand = np.random.normal(mean * seasonal_patterns[i][t], std_dev)
            demand[t, i] = max(0, int(base_demand))

    return demand

def generate_complex_seasonal_pattern(length, amplitude_range, period_range, num_periods_range):
    t = np.arange(length)
    seasonal_pattern = np.zeros(length)
    num_periods = np.random.randint(*num_periods_range)
    period_length = length // num_periods

    for _ in range(num_periods):
        frequency = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(0, 2 * np.pi)
        start = np.random.randint(0, length - period_length)
        end = start + period_length
        seasonal_pattern[start:end] += amplitude * np.sin(frequency * 2 * np.pi * t[start:end] / period_length + phase)

    seasonal_pattern += np.random.normal(scale=0.1, size=length)
    return np.maximum(1, seasonal_pattern)

def generate_random_dropoffs(num_nodes, demand):
    dropoffs = {}
    for t in range(demand.shape[0]):
        for node in range(demand.shape[1]):
            num_pickups = demand[t, node]
            dropoff_list = [random.randint(0, num_nodes - 1) for _ in range(num_pickups)]
            dropoffs[(t, node)] = dropoff_list
    return dropoffs    

def generate_driver_series(num_nodes, length, mean, std_dev):
    driver_counts = np.random.normal(loc=mean, scale=std_dev, size=(length, num_nodes))
    driver_counts = np.clip(driver_counts, 0, None)  # Ensure non-negative counts
    return np.round(driver_counts).astype(int)


def visualize_pickup_series(demand):
    length, num_nodes = demand.shape
    time_steps = np.arange(length)
    
    plt.figure(figsize=(6, 4))
    for i in range(num_nodes):
        plt.plot(time_steps, demand[:, i], label=f'Node {i}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Pickups')
    plt.title('Taxi Pickup Time Series')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
    plt.grid(True)
    plt.show()

def visualize_node_influence(demand):
    correlation_matrix = np.corrcoef(demand.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=True, yticklabels=True)
    plt.title('Correlation Heatmap of Taxi Pickup Time Series')
    plt.xlabel('Node')
    plt.ylabel('Node')
    plt.show()

def calculate_distance(graph, start, end):
    try:
        path_length = nx.shortest_path_length(graph, source=start, target=end, weight='weight', method='dijkstra')
    except nx.NetworkXNoPath:
        path_length = np.inf
    return path_length

# PRINTING SAMPLES
def printSampleSeries(pickup_series, drivers_series, dropoffs):    
    print("Sample pickup series for Node 0:", pickup_series[:, 0])
    print("Sample driver series for Node 0:", drivers_series[:, 0])
    print("Dropoffs for first time step at Node 0:", dropoffs.get((0, 0), "No pickups"))




# GRAPH GENERATION

num_nodes = 10  # Define the number of nodes in the grid
skip_prob = 0.15  # Probability of skipping an edge between nodes
extra_edges = 0.15  # Additional edges as a fraction of total nodes

adj_matrix = generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)


# DEMAND SIMULATION

length = 100  # Duration 
mean = 10  # Mean demand per node 
std_dev = 3  # Demand stdev
amplitude_range = (0.5, 1.5)  # Amplitude demand range
period_range = (10, 50)  # Range of demand periods
num_periods_range = (3, 10)  # Range for number of periods within length

# TIME SERIES GENERATION

pickup_series = generate_seasonal_demand(length, num_nodes, mean, std_dev, amplitude_range, period_range, num_periods_range)
dropoffs = generate_random_dropoffs(num_nodes, pickup_series)

# DRIVER SIMULATION

driver_mean = 10 
driver_std_dev = 2 
drivers_series = generate_driver_series(num_nodes, length, driver_mean, driver_std_dev)


# Example usage of the function
d = 10  # Define the critical time period