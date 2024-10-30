import numpy as np
import random
from scipy.sparse.csgraph import shortest_path

# paper: https://lbsresearch.london.edu/id/eprint/2475/1/ALI2%20Dynamic%20Stochastic.pdf

def obtain_tildex(flow_matrix, i, print_stuff = False):
    """
    Extract the flow rates \tilde{x}_{i,j} for a specific node i from the flow matrix.
    
    :param flow_matrix: The flow matrix from the QB optimization (with flows \tilde{x}_{i,j})
    :param i: The specific node i (Active Driver) for which to extract flows
    :return: A vector of flows \tilde{x}_{i,j} for the given node i.
    """
    # Extract the flow rates for node i (row i of the flow matrix)
    tildex_i_j = flow_matrix[i, :]
    if print_stuff:
        print("Tilde", tildex_i_j)
    return tildex_i_j

def generate_label(is_rider, node, flow_matrix, tildex_i_j, riders, drivers, lambda_i, lambda_j, mu_i, print_stuff=True):
    if is_rider:
        return 0

    num_riders = len(flow_matrix[0])
    num_drivers = len(flow_matrix)

    if print_stuff: 
        print(f"\nGenerating label for driver at node {node}")
        print(f"tildex_i_j: {tildex_i_j}")

    S_i = [j for j in range(num_riders) if tildex_i_j[j] > 0]
    if print_stuff: 
        print(f"S_i: {S_i}")

    if not S_i:
        if print_stuff:
            print(f"No valid matches for Driver at node {node}. Assigning default label.")
        return -1

    # Calculate lambda^p_j for each j in S_i
    lambdap = {}
    for j in S_i:
        lambdap[j] = tildex_i_j[j] + sum(flow_matrix[i][j] for i in range(num_drivers))
    if print_stuff: 
        print(f"lambdap: {lambdap}")

    # Sort S_i in descending order based on tildex_i_j[j] / lambdap[j]
    S_i.sort(key=lambda j: (tildex_i_j[j] / lambdap[j], random.random()), reverse=True)
    if print_stuff: 
        print(f"Sorted S_i: {S_i}")

    # Initialize hatlambda_il for each j in S_i
    hatlambda_il = np.zeros(len(S_i))

    # Base case: Compute \hat{\lambda}_{i,|S_i|}
    last_index = S_i[-1]
    hatlambda_il[-1] = (mu_i + sum(lambdap[j] for j in S_i)) / lambdap[last_index] * tildex_i_j[last_index]

    # Inductive step: Compute \hat{\lambda}_{i,\ell} for remaining elements
    for idx in range(len(S_i) - 2, -1, -1):
        j_l = S_i[idx]

        # Calculate remaining \hat{\lambda} based on previous values
        remaining_hatlambda = sum(
            (lambdap[S_i[q]] / (mu_i + sum(lambdap[k] for k in S_i[idx+1:])))
            * hatlambda_il[q]
            for q in range(idx+1, len(S_i))
        )

        # Calculate \hat{\lambda}_{i,\ell} using the induction formula
        hatlambda_il[idx] = (mu_i + sum(lambdap[k] for k in S_i[idx:])) / lambdap[j_l] * \
                            (tildex_i_j[j_l] - remaining_hatlambda)

        if print_stuff: 
            print(f"Step {idx}: hatlambda_il = {hatlambda_il}")

    if print_stuff: 
        print(f"Final hatlambda_il before normalization: {hatlambda_il}")

    # Normalize \hat{\lambda}_{i,\ell}
    sum_hatlambda = np.sum(hatlambda_il)
    if sum_hatlambda <= 0:
        if print_stuff: 
            print(f"Warning: Sum of hatlambda_il is non-positive. Assigning default label.")
        return -1

    normalized_hatlambda_il = hatlambda_il / sum_hatlambda
    if print_stuff: 
        print(f"Normalized hatlambda_il: {normalized_hatlambda_il}")

    # Sample the label based on the normalized distribution
    chosen_label = np.random.choice(S_i, p=normalized_hatlambda_il)
    if print_stuff: 
        print(f"Chosen label: {chosen_label}")

    return chosen_label


def generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob=0.15, extra_edges=0.15):
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0")

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Create self-loops
    for i in range(num_nodes):
        adjacency_matrix[i][i] = 1  # Self-loop at each node

    # Create edges between adjacent nodes
    for i in range(num_nodes - 1):
        if random.random() > skip_prob:
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

    # Add additional random edges based on extra_edges probability
    num_extra_edges = int(extra_edges * num_nodes)
    edges_added = 0
    while edges_added < num_extra_edges:
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node1 != node2 and adjacency_matrix[node1][node2] == 0:
            adjacency_matrix[node1][node2] = 1
            adjacency_matrix[node2][node1] = 1
            edges_added += 1
    
    print("Adjacency matrix:", adjacency_matrix)
    return adjacency_matrix

def adjacency_to_rewards(adjacency_matrix, reward_value=8, distance_penalty=2):
    num_nodes = adjacency_matrix.shape[0]
    rewards = {}

    # Calculate shortest paths between all pairs of nodes
    dist_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)

    active_types = [f"Active Driver Node {i}" for i in range(num_nodes)]
    passive_types = [f"Passive Rider Node {i}" for i in range(num_nodes)]

    for i, active_type in enumerate(active_types):
        rewards[active_type] = {}
        for j, passive_type in enumerate(passive_types):
            distance = int(dist_matrix[i][j])
            rewards[active_type][passive_type] = max(reward_value -distance_penalty * distance, 0)  # Subtract distance from reward
            
    return rewards

class RealizationGraph: 
    def __init__(self):
        self.active_drivers = []
        self.passive_riders = []
        self.total_trip_distance = 0
        self.total_wait_time = 0
        self.total_rewards = 0
        self.num_drivers_matched = 0
        self.num_riders_matched = 0
        self.total_rider_count = 0
        self.total_driver_count = 0
        self.print = False

    def add_driver(self, driver):
        if self.print: 
            print(f"Driver added at location {driver.location}")
        self.active_drivers.append(driver)
        self.total_driver_count += 1  # Count every driver added

    def remove_driver(self, driver):
        if self.print: 
            print(f"Driver removed at location {driver.location}")
        self.active_drivers.remove(driver)

    def add_rider(self, rider):
        if self.print: 
            print(f"Rider added at location {rider.location}")
        self.passive_riders.append(rider)
        self.total_rider_count += 1  # Count every rider added

    def remove_rider(self, rider):
        if self.print: 
            print(f"Rider removed at location {rider.location}")
        self.passive_riders.remove(rider)

    def find_driver_for_rider(self, rider, rewards):
        if self.active_drivers:
            matched_driver = self.active_drivers.pop(0)  # Pop the first available driver

            wait_time = rider.arrival_time - matched_driver.arrival_time
            trip_distance = abs(rider.location - matched_driver.location)
            reward = rewards[matched_driver.type][rider.type]  # Get reward for this match

            if self.print: 
                print(f"Matching Rider at {rider.location} with Driver at {matched_driver.location}, Reward: {reward}")
            
            self.total_wait_time += wait_time
            self.total_trip_distance += trip_distance
            self.total_rewards += reward  # Reward is added here
            self.num_drivers_matched += 1
            self.num_riders_matched += 1

            return matched_driver
        return None

    def print_summary(self):
        if self.num_riders_matched > 0:
            average_trip_distance = self.total_trip_distance / self.num_riders_matched
            average_wait_time = self.total_wait_time / self.num_riders_matched
            average_reward = self.total_rewards / self.num_riders_matched
        else:
            average_trip_distance = 0
            average_wait_time = 0
            average_reward = 0

        print("\nSummary Statistics:")
        print(f"Number of Drivers Matched: {self.num_drivers_matched}")
        print(f"Number of Riders Matched: {self.num_riders_matched}")
        print(f"Total Riders Processed: {self.total_rider_count}")
        print(f"Total Drivers Processed: {self.total_driver_count}")
        print(f"Total Rewards: {self.total_rewards:.2f}")
        print(f"Average Reward per Transaction: {average_reward:.2f}")
        print(f"Average Trip Distance: {average_trip_distance:.2f} units")
        print(f"Average Wait Time: {average_wait_time:.2f} units")