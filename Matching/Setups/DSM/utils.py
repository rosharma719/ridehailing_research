import numpy as np
import random

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

def adjacency_to_rewards(adjacency_matrix):
    """
    Converts the adjacency matrix into a reward dictionary suitable for Gurobi input.
    
    :param adjacency_matrix: A 2D numpy array representing the adjacency matrix of the graph.
    :return: A dictionary of rewards where each key is a type pair (e.g., 'Active Rider 0' to 'Passive Driver 1'),
             and the value is the reward based on adjacency (1 if adjacent, 0 otherwise).
    """
    num_nodes = adjacency_matrix.shape[0]
    rewards = {}

    # Create distinct types for each node as 'Active' (drivers) and 'Passive' (riders)
    active_types = [f"Active Driver {i}" for i in range(num_nodes)]
    passive_types = [f"Passive Rider {i}" for i in range(num_nodes)]

    # Populate the rewards based on adjacency
    for i, active_type in enumerate(active_types):
        rewards[active_type] = {}
        for j, passive_type in enumerate(passive_types):
            # Reward is 1 if there's an edge between the corresponding nodes, otherwise 0
            rewards[active_type][passive_type] = 1 if adjacency_matrix[i][j] == 1 else 0

    return rewards

