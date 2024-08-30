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
    print(adjacency_matrix)
    return adjacency_matrix

def adjacency_to_rewards(adjacency_matrix, reward_value=8):
    num_nodes = adjacency_matrix.shape[0]
    rewards = {}

    # Create distinct types for each node as 'Active' (drivers) and 'Passive' (riders)
    active_types = [f"Active Driver Node {i}" for i in range(num_nodes)]
    passive_types = [f"Passive Rider Node {i}" for i in range(num_nodes)]

    # Populate the rewards based on the sum of edges between driver and rider nodes
    for i, active_type in enumerate(active_types):
        rewards[active_type] = {}
        for j, passive_type in enumerate(passive_types):
            if adjacency_matrix[i][j] == 1:
                # Reward is 8 minus the sum of edges connected to both the driver and rider nodes
                edge_sum = np.sum(adjacency_matrix[i]) + np.sum(adjacency_matrix[j])
                rewards[active_type][passive_type] = reward_value-edge_sum
            elif i == j:
                edge_sum = np.sum(adjacency_matrix[i]) + np.sum(adjacency_matrix[j])
                rewards[active_type][passive_type] = reward_value-edge_sum 
            else:
                # No reward for nodes that are not directly connected
                rewards[active_type][passive_type] = 0

    return rewards

class RealizationGraph: 
    def __init__(self):
        self.active_drivers = []
        self.passive_riders = []
        self.total_trip_distance = 0
        self.num_drivers_matched = 0
        self.num_riders_matched = 0
        self.total_rider_count = 0
        self.print = False

    def add_driver(self, driver):
        if self.print: 
            print(f"Driver added at location {driver.location}")
        self.active_drivers.append(driver)

    def remove_driver(self, driver):
        if self.print: 
            print(f"Driver removed at location {driver.location}")
        self.active_drivers.remove(driver)

    def add_rider(self, rider):
        if self.print: 
            print(f"Rider added at location {rider.location}")
        self.passive_riders.append(rider)
        self.total_rider_count += 1  # Ensure this is counted every time a rider arrives

    def remove_rider(self, rider):
        if self.print: 
            print(f"Rider removed at location {rider.location}")
        self.passive_riders.remove(rider)

    def find_driver_for_rider(self, rider):
        if self.active_drivers:
            matched_driver = self.active_drivers.pop(0)  # Pop the first available driver

            trip_distance = abs(rider.location - matched_driver.location)

            if self.print: 
                print(f"Matching Rider at {rider.location} with Driver at {matched_driver.location}")
            
            self.total_trip_distance += trip_distance
            self.num_drivers_matched += 1
            self.num_riders_matched += 1

            return matched_driver
        return None

    def print_summary(self):
        if self.num_riders_matched > 0:
            average_trip_distance = self.total_trip_distance / self.num_riders_matched
        else:
            average_trip_distance = 0

        print("\nSummary Statistics:")
        print(f"Number of Drivers Matched: {self.num_drivers_matched}")
        print(f"Number of Riders Matched: {self.num_riders_matched}")
        print(f"Total Riders Processed: {self.total_rider_count}")
        print(f"Average Trip Distance: {average_trip_distance:.2f} units")