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

def adjacency_to_rewards(adjacency_matrix, abandonment_cost=-10):
    """
    Converts the adjacency matrix into a reward dictionary suitable for Gurobi input,
    using -#edges as the cost of each match and a configurable abandonment cost.
    
    :param adjacency_matrix: A 2D numpy array representing the adjacency matrix of the graph.
    :param abandonment_cost: The cost of abandonment, default set to -10.
    :return: A dictionary of rewards where each key is a type pair (e.g., 'Active Driver 0' to 'Passive Rider 1'),
             and the value is the negative edge count for adjacency, or abandonment cost.
    """
    num_nodes = adjacency_matrix.shape[0]
    rewards = {}

    # Create distinct types for each node as 'Active' (drivers) and 'Passive' (riders)
    active_types = [f"Active Driver {i}" for i in range(num_nodes)]
    passive_types = [f"Passive Rider {i}" for i in range(num_nodes)]
    
    # Add dummy type for abandonment
    dummy_type = "Dummy Rider"

    # Populate the rewards based on adjacency and abandonment costs
    for i, active_type in enumerate(active_types):
        rewards[active_type] = {}
        for j, passive_type in enumerate(passive_types):
            # Use negative edge count as cost for direct connection
            rewards[active_type][passive_type] = -np.sum(adjacency_matrix[i]) if adjacency_matrix[i][j] == 1 else abandonment_cost

        # Set abandonment penalty for this driver to the dummy rider
        rewards[active_type][dummy_type] = abandonment_cost

    # Add the dummy rider in the list of passive types
    passive_types.append(dummy_type)

    return rewards


def calculate_label_distribution(flow_matrix, active_types, passive_types):
    """
    Calculate the label distribution for each type based on the flow matrix.
    
    :param flow_matrix: A NumPy array representing the flow matrix of optimal match rates.
    :param active_types: List of active types.
    :param passive_types: List of passive types.
    :return: A dictionary with the probability distribution of passive labels for each type.
    """
    label_distribution = {}
    for j_idx, j in enumerate(passive_types):
        # Calculate passive arrival rate λ_p_i
        lambda_p_j = np.sum(flow_matrix[:, j_idx])

        # Calculate the total arrival rate λ_i
        total_lambda_j = np.sum(flow_matrix[:, j_idx]) + np.sum(flow_matrix[j_idx, :])

        # Probability distribution of passive label
        prob_passive = lambda_p_j / total_lambda_j if total_lambda_j > 0 else 0
        label_distribution[j] = prob_passive
    
    return label_distribution