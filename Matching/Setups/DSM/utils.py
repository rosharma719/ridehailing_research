import numpy as np
import random
from scipy.sparse.csgraph import shortest_path

# paper: https://lbsresearch.london.edu/id/eprint/2475/1/ALI2%20Dynamic%20Stochastic.pdf

def obtain_tildex():
    # solve optimization to obtain tilde{x} in (23) of the paper 

# node (type = (rider/drive, node) pair is called type)
# \mathcal{T} <- set of types, which includes driver and rider both
def generate_label(is_rider, node, tildex_ij, tildex_i):
    # tildex is (\tilde{x}_{i,j})_{i,j} in the paper, assuming two-dim array
    # If rider -> label 0 (passive, always matches the precedent driver) 
    # else - driver -> random labeling 
    # (in the case of *two nodes*, the type is 1 or 2 - label 2 matches to both nodes, where label 1 matches to rider of the same node only) 
    labels = []
    lambdap = [] # (\lambda_p^j)_j in the paper
    for j in riders:
        lambdap[j] = tildex_i[j] + sum of tildex_ij[i][j]
    for j in riders:
        labels[j] = 0 # trivial labeling
    for i in drivers:
        S_i = # set of js such that tildex[i][j]> 0
        hatlambda_il = np.zeros(S_i) # \hat{\lambda}_{i,p} in the paper
        priorities = # elements in S_i's value of tildex[i][j]/\lambdap[j]
        # Sort S_i in descending order of priorities. For example, for node 1, S_i should be [1, 2]
        Use Eq. (24) to obtain hatlambda_il[len(S_i)-1] # the last index 
        for j in reversed(S_i)[1:]: 
            induction step to obtain \hat{\lambda}_{i,len(S_i)-2}, \hat{\lambda}_{i,len(S_i)-3},...,\hat{\lambda}_{i,0}, denoted in Eq. (25) 
        normalized_hatlambda_il # driver_s distribution of labels, normalized to sum 1 (Eq. (27))

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
    
    print(adjacency_matrix)
    return adjacency_matrix

def adjacency_to_rewards(adjacency_matrix, reward_value=8):
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
            if distance >= 0:  # Only valid distances
                rewards[active_type][passive_type] = max(reward_value - distance, 0)  # Subtract distance from reward
            else:
                rewards[active_type][passive_type] = 0  # No valid path, no reward

    print("Rewards Matrix:")
    print(rewards, '\n\n')
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

