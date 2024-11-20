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


def generate_label(is_rider, node, results, lambda_i, lambda_j, mu_i, print_stuff=False):
    if is_rider:
        return 0

    flow_matrix = results['flow_matrix']
    abandonment_rates = results['abandonment']
    passive_unmatched = results['passive_unmatched']
    num_riders = flow_matrix.shape[1]
    tildex_i_j = flow_matrix[node, :]
    S_i = [j for j in range(num_riders) if tildex_i_j[j] > 0]
    
    if not S_i:
        return -1

    driver_name = f"Active Driver Node {node}"
    x_a_i = abandonment_rates[driver_name]
    sojourn_rate = mu_i[driver_name]

    lambdap = {}
    for j in S_i:
        rider_name = f"Passive Rider Node {j}"
        x_j = passive_unmatched[rider_name]
        lambdap_j = x_j + sum(flow_matrix[:, j])
        lambdap[j] = lambdap_j

    S_i.sort(key=lambda j: tildex_i_j[j] / lambdap[j], reverse=True)

    hatlambda_il = np.zeros(len(S_i))
    sum_lambdap_Si = sum(lambdap[j] for j in S_i)

    L = len(S_i) - 1
    j_L = S_i[L]
    numerator = sojourn_rate + sum_lambdap_Si
    denominator = lambdap[j_L]
    hatlambda_il[L] = (numerator / denominator) * tildex_i_j[j_L]
    
    if print_stuff:
        print(f"Base Case - L = {L}, j_L = {j_L}")
        print(f"  sojourn_rate: {sojourn_rate}")
        print(f"  sum_lambdap_Si: {sum_lambdap_Si}")
        print(f"  numerator: {numerator}")
        print(f"  denominator: {denominator}")
        print(f"  hatlambda_il[{L}]: {hatlambda_il[L]}")

    tolerance = 1e-6
    for idx in range(L - 1, -1, -1):
        j_l = S_i[idx]
        sum_lambdap_k = sum(lambdap[S_i[k]] for k in range(idx, L + 1))

        numerator = sojourn_rate + sum_lambdap_k
        denominator = lambdap[j_l]
        inner_sum = 0

        for m_pos in range(idx + 1, L + 1):  
            m = S_i[m_pos]
            
            # Compatibility set S_i[m]
            compatibility_set = [j for j in S_i]  
            
            sum_lambdap_m_sub = sum(lambdap[j] for j in compatibility_set)

            denominator_m = sojourn_rate + sum_lambdap_m_sub
            fraction = (lambdap[m] / denominator_m) * hatlambda_il[m_pos]

            fraction = 0 if abs(fraction) < tolerance else fraction
            inner_sum += fraction

            if print_stuff:
                print(f"  Inner Loop - idx = {idx}, m_pos = {m_pos} (node {m})")
                print(f"    compatibility_set: {compatibility_set}")
                print(f"    sum_lambdap_m_sub: {sum_lambdap_m_sub}")
                print(f"    denominator_m: {denominator_m}")
                print(f"    fraction: {fraction}")
                print(f"    inner_sum: {inner_sum}")

        hatlambda_il[idx] = (numerator / denominator) * (tildex_i_j[j_l] - inner_sum)
        hatlambda_il[idx] = 0 if abs(hatlambda_il[idx]) < tolerance else hatlambda_il[idx]
        
        if print_stuff:
            print(f"Inductive Step - idx = {idx}, j_l = {j_l}")
            print(f"  sum_lambdap_k: {sum_lambdap_k}")
            print(f"  numerator: {numerator}")
            print(f"  denominator: {denominator}")
            print(f"  tildex_i_j[j_l]: {tildex_i_j[j_l]}")
            print(f"  inner_sum: {inner_sum}")
            print(f"  hatlambda_il[{idx}]: {hatlambda_il[idx]}")

    if np.any(hatlambda_il < 0):
        if print_stuff:
            print("Negative probabilities encountered in hatlambda_il:", hatlambda_il)
        raise ValueError("Calculated negative probabilities in label generation.")

    sum_hatlambda = np.sum(hatlambda_il)
    if sum_hatlambda <= 0:
        if print_stuff:
            print("Sum of probabilities is non-positive, defaulting label.")
        return -1

    normalized_hatlambda_il = hatlambda_il / sum_hatlambda
    if print_stuff:
        print("Label distribution (normalized probabilities):", normalized_hatlambda_il)

    chosen_label = np.random.choice(S_i, p=normalized_hatlambda_il)

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
        
    def get_waiting_counts_by_node(self):
        """
        Returns a dictionary with the counts of riders and drivers waiting at each node.
        """
        counts = {}
        for rider in self.passive_riders:
            counts.setdefault(rider.location, {"riders": 0, "drivers": 0})
            counts[rider.location]["riders"] += 1

        for driver in self.active_drivers:
            counts.setdefault(driver.location, {"riders": 0, "drivers": 0})
            counts[driver.location]["drivers"] += 1

        return counts

    def get_waiting_riders_at_node(self, node):
        """
        Returns a list of all riders waiting at a specific node.
        """
        return [rider for rider in self.passive_riders if rider.location == node]

    def get_waiting_drivers_at_node(self, node):
        """
        Returns a list of all drivers waiting at a specific node.
        """
        return [driver for driver in self.active_drivers if driver.location == node]