import numpy as np
import random
from scipy.sparse.csgraph import shortest_path
from eventgenerator import *
from matplotlib import pyplot as plt
import math
from math import ceil, sqrt
import networkx as nx


# paper: https://lbsresearch.london.edu/id/eprint/2475/1/ALI2%20Dynamic%20Stochastic.pdf

def mean(x): return sum(x)/len(x) if x else 0.0

def list_to_named_dict(rate_list, prefix):
    """
    Converts a list of rates into a dictionary with keys like 'Active Driver Node 0'.

    Parameters:
    - rate_list: list of numerical values
    - prefix: string to prepend to each node (e.g., 'Active Driver Node', 'Passive Rider Node')

    Returns:
    - dict mapping names to values
    """
    return {f"{prefix} {i}": rate for i, rate in enumerate(rate_list)}


def generate_param_normal_distribution(node_names, mean, std_dev, min_val=0.01, max_val=1.0, decimals=2, seed=None):
    """
    Generate a dictionary of mu_i values (abandonment rates) normally distributed,
    bounded and rounded to the nearest specified decimal.

    Parameters:
    - node_names: list of string keys for the nodes (e.g., ["Active Driver Node 0", ...])
    - mean: mean of the normal distribution
    - std_dev: standard deviation of the normal distribution
    - min_val: minimum value to clip to (default 0.01)
    - max_val: maximum value to clip to (default 1.0)
    - decimals: number of decimal places to round to (default 2)
    - seed: random seed for reproducibility (optional)

    Returns:
    - Dictionary: {node_name: mu_i_value}
    """
    if seed is not None:
        np.random.seed(seed)

    mu_values = np.random.normal(loc=mean, scale=std_dev, size=len(node_names))
    mu_values = np.clip(mu_values, min_val, max_val)
    mu_values = np.round(mu_values, decimals)

    return {name: mu for name, mu in zip(node_names, mu_values)}


def count_events(event_queue):
    """
    Count the number of arrival and abandonment events in the event queue.

    :param event_queue: The event queue containing events.
    :return: A tuple (num_arrivals, num_abandonments) representing the counts.
    """
    num_arrivals = 0
    num_abandonments = 0

    temp_queue = event_queue.clone()  # Clone to preserve original queue order

    while not temp_queue.is_empty():
        event = temp_queue.get_next_event()
        if event.event_type == 'arrival':
            num_arrivals += 1
        elif event.event_type == 'abandonment':
            num_abandonments += 1

    return num_arrivals, num_abandonments


def perform_match(realization_graph, driver, rider, rewards, distance_matrix, event_queue):
    """
    Perform a match between a driver and a rider, updating the realization graph, rewards,
    and removing associated abandonment events.
    """
    print_stuff = False
    try:
        if print_stuff: 
            print("List of active drivers:")
            for driver in realization_graph.active_drivers: 
                print("Driver", driver.arrival_time)
            print("\n")

        # Check if the driver and rider are in the system
        if driver not in realization_graph.active_drivers:
            if print_stuff: 
                print(f"ERROR: Driver {driver.arrival_time:.3f} not found in active_drivers.")
                print(f"Current active drivers: {[d.arrival_time for d in realization_graph.active_drivers]}")
            raise RuntimeError(f"Driver not found in active_drivers: {driver}")

        if rider not in realization_graph.passive_riders:
            if print_stuff: 
                print(f"ERROR: Rider {rider.arrival_time:.3f} not found in passive_riders.")
                print(f"Current passive riders: {[r.arrival_time for r in realization_graph.passive_riders]}")
            raise RuntimeError(f"Rider not found in passive_riders: {rider}")

        # Calculate match statistics
        wait_time = abs(driver.arrival_time - rider.arrival_time)
        trip_distance = distance_matrix[driver.location][rider.location]
        reward = rewards[driver.type][rider.type]

        # Increment system-wide counters
        realization_graph.total_wait_time += wait_time
        realization_graph.total_trip_distance += trip_distance
        realization_graph.total_rewards += reward
        realization_graph.num_drivers_matched += 1
        realization_graph.num_riders_matched += 1

        # Remove associated abandonment events
        remove_abandonment_event(event_queue, driver)
        remove_abandonment_event(event_queue, rider)

        # Remove the driver and rider from the system only after all operations are complete
        realization_graph.remove_driver(driver)
        realization_graph.remove_rider(rider)

        # Log the match
        if print_stuff: 
            print(
                f"Matched Driver {driver.arrival_time:.3f} at {driver.location} with "
                f"Rider {rider.arrival_time:.3f} at {rider.location}, "
                f"Reward: {reward:.3f}, Trip Distance: {trip_distance:.3f}, Wait Time: {wait_time:.3f}."
            )

    except ValueError as e:
        raise RuntimeError(
            f"Inconsistent state during match removal: {e}. Driver: {driver}, Rider: {rider}"
        )



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
        if print_stuff:
            print(f"Node {node} is a rider. Returning label 0.")
        return 0, {}

    flow_matrix = results['flow_matrix']
    passive_unmatched = results['passive_unmatched']
    num_riders = flow_matrix.shape[1]

    if print_stuff:
        print(f"\nNODE {node} arrives")

    tildex_i_j = flow_matrix[node, :]
    S_i = [j for j in range(num_riders) if tildex_i_j[j] > 1e-8]

    if not S_i:
        if print_stuff:
            print(f"No compatibility set found for Node {node}. Returning -1.")
        return -1, {}

    driver_name = f"Active Driver Node {node}"
    sojourn_rate = mu_i[driver_name]

    # Compute λ^p_j for each rider in S_i
    lambdap = {
        j: passive_unmatched[f"Passive Rider Node {j}"] + sum(flow_matrix[:, j])
        for j in S_i
    }

    # === Sort S_i and corresponding flows ===
    sort_key = lambda j: (tildex_i_j[j] / lambdap[j], random.random())
    S_i_sorted = sorted(S_i, key=sort_key, reverse=True)
    sorted_flows = np.array([tildex_i_j[j] for j in S_i_sorted])

    if print_stuff:
        print(f"Sorted Compatibility Set S_i: {S_i_sorted}")
        print(f"Sorted Flows: {sorted_flows}")

    # === Initialize hatlambda_il ===
    hatlambda_il = np.zeros(len(S_i_sorted))
    sum_lambdap_Si = sum(lambdap[j] for j in S_i_sorted)

    # === Base case ===
    L = len(S_i_sorted) - 1
    j_L = S_i_sorted[L]
    numerator = sojourn_rate + sum_lambdap_Si
    denominator = lambdap[j_L]
    hatlambda_il[L] = (numerator / denominator) * sorted_flows[L]

    if print_stuff:
        print(f"Base Case (L = {L}, j_L = {j_L}):")
        print(f"  Numerator: {numerator}")
        print(f"  Denominator: {denominator}")
        print(f"  Flow: {sorted_flows[L]}")
        print(f"  Hatlambda_il[{L}]: {hatlambda_il[L]}")

    # === Recursive case ===
    for idx in range(L - 1, -1, -1):
        j_l = S_i_sorted[idx]
        numerator = sojourn_rate + sum(lambdap[j] for j in S_i_sorted[idx:])
        denominator = lambdap[j_l]
        current_flow = sorted_flows[idx]
        next_flow = sorted_flows[idx + 1]

        hatlambda_il[idx] = (numerator / denominator) * (current_flow - next_flow)

        if print_stuff:
            print(f"Recursive Case (idx = {idx}, j_l = {j_l}):")
            print(f"  Numerator: {numerator}")
            print(f"  Denominator: {denominator}")
            print(f"  Current flow: {current_flow}")
            print(f"  Next flow: {next_flow}")
            print(f"  Hatlambda_il[{idx}]: {hatlambda_il[idx]}")

    # === Clean and Normalize ===
    hatlambda_il = np.maximum(hatlambda_il, 0)
    sum_hatlambda = np.sum(hatlambda_il)
    if sum_hatlambda == 0:
        return -1, {}

    normalized_hatlambda_il = hatlambda_il / sum_hatlambda

    if print_stuff:
        print(f"Normalized Hatlambda_il: {normalized_hatlambda_il}")

    # === Sample a label ===
    chosen_label_index = np.random.choice(range(len(S_i_sorted)), p=normalized_hatlambda_il)
    chosen_label = S_i_sorted[chosen_label_index]

    if print_stuff:
        print(f"Chosen Label: {chosen_label}")

    # === Build label → compatibility set map ===
    label_to_set_map = {
        S_i_sorted[idx]: S_i_sorted[: idx + 1] for idx in range(len(S_i_sorted))
    }
    label_to_set_map[-1] = []

    if print_stuff:
        print("LABEL TO SET MAP")
        print(label_to_set_map)
        print(f"Label {chosen_label}'s Compatibility Set: {label_to_set_map[chosen_label]}")

    return chosen_label, label_to_set_map


def generate_grid_adjacency_matrix(num_nodes):
    """
    Generates a grid-like adjacency matrix for a given number of nodes.
    
    Args:
        num_nodes (int): The total number of nodes.
    
    Returns:
        np.ndarray: The adjacency matrix for the grid-like structure.
    """
    if num_nodes <= 1:
        raise ValueError("Number of nodes must be greater than 1")
    
    # Calculate the optimal number of rows and columns
    rows = int(sqrt(num_nodes))
    cols = ceil(num_nodes / rows)
    
    # Create the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        # Add self-loops
        adjacency_matrix[i][i] = 1

        # Connect to the right neighbor if it exists
        if (i + 1) % cols != 0 and i + 1 < num_nodes:
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

        # Connect to the bottom neighbor if it exists
        if i + cols < num_nodes:
            adjacency_matrix[i][i + cols] = 1
            adjacency_matrix[i + cols][i] = 1

    # Plot the grid for visualization (optional)
    #plot_grid_graph(adjacency_matrix, rows, cols)
    
    return adjacency_matrix  # Return only the adjacency matrix



def plot_grid_graph(adj_matrix, rows, cols):
    """
    Visualizes the grid-like graph for a given adjacency matrix.
    
    Args:
        adj_matrix (np.ndarray): The adjacency matrix.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = {i: (i % cols, -i // cols) for i in range(len(adj_matrix))}
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    plt.title(f"Grid with {len(adj_matrix)} Nodes ({rows}x{cols})")
    plt.show()


def generate_ring_adjacency_matrix(num_nodes):
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0")

    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Add self-loops
    for i in range(num_nodes):
        adjacency_matrix[i][i] = 1

    # Create bidirectional edges in a ring topology
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes  # Connects in a circular fashion
        adjacency_matrix[i][next_node] = 1
        adjacency_matrix[next_node][i] = 1


    #print("RING\n\n", adjacency_matrix)
    #plot_ring_graph(adjacency_matrix)
    return adjacency_matrix


def plot_ring_graph(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): 
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=12)
    plt.title("Ring Topology")
    plt.show()



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

def divided_adjacency_to_rewards(adjacency_matrix, reward_value=8, denom_multiple=1):
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
            rewards[active_type][passive_type] =  reward_value/((distance+1)*denom_multiple) 
            
    return rewards

def increasing_distance_penalty(adjacency_matrix, reward_value=8): 
    num_nodes = adjacency_matrix.shape[0]
    rewards = {}

    dist_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)


    active_types = [f"Active Driver Node {i}" for i in range(num_nodes)]
    passive_types = [f"Passive Rider Node {i}" for i in range(num_nodes)]


    for i, active_type in enumerate(active_types):
        rewards[active_type] = {}
        for j, passive_type in enumerate(passive_types):
            distance = int(dist_matrix[i][j])
            rewards[active_type][passive_type] = reward_value-2*distance  # Subtract 2*distance from reward
            
    print("REWARDS", rewards)
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
        self.print = False  # Enable or disable print logs

    def add_driver(self, driver):
        if self.print: 
            print(f"Driver {driver.arrival_time:.3f} added at location {driver.location}.")
        self.active_drivers.append(driver)
        
        if self.print:  
            print(f"Current active drivers: {[d.arrival_time for d in self.active_drivers]}")

        self.total_driver_count += 1  # Count every driver added

        if driver not in self.active_drivers:
            if self.print:
                print(f"ERROR: Driver {driver.arrival_time:.3f} not found in active_drivers AFTER ADDING.")
            raise RuntimeError(f"Driver not found in active_drivers: {driver}")

       

    def remove_driver(self, driver):
        if driver not in self.active_drivers:
            print(f"WARNING: Attempted to remove non-existent driver {driver.arrival_time:.3f} from active_drivers.")
            return
        
        if self.print:
            print(f"Driver {driver.arrival_time:.3f} removed at location {driver.location}.")
        self.active_drivers.remove(driver)


    def add_rider(self, rider):
        if self.print: 
            print(f"Rider {rider.arrival_time:.3f} added at location {rider.location}.")
        self.passive_riders.append(rider)
        self.total_rider_count += 1  # Count every rider added

    def remove_rider(self, rider):
        if self.print: 
            print(f"Rider {rider.arrival_time:.3f} removed at location {rider.location}.")
        self.passive_riders.remove(rider)

    def find_rider_for_driver(self, driver, rewards, distance_matrix):
        """
        Find the most compatible rider for a driver based on compatibility sets and rewards.
        Searches across all waiting riders in the system.

        :param driver: The arriving driver.
        :param rewards: The reward matrix.
        :param distance_matrix: The shortest path distance matrix between all nodes.
        :return: The matched rider, or None if no match is found or reward is ≤ 0.
        """
        if self.print:
            print(f"Current waiting riders: {[r.arrival_time for r in self.passive_riders]}")

        if not self.passive_riders:
            return None

        matched_rider = None
        max_reward = -float('inf')  # Start with the lowest possible reward
        best_trip_distance = float('inf')  # For tie-breaking based on trip distance

        for rider in self.passive_riders:
            # Check if the rider's location is in the driver's compatibility set
            if rider.location in driver.compatibility_set:
                # Calculate reward and trip distance
                reward = rewards[driver.type][rider.type]
                trip_distance = distance_matrix[driver.location][rider.location]

                # Skip if reward is non-positive
                if reward <= 0:
                    continue

                # Update the matched rider if this one is better
                if reward > max_reward or (reward == max_reward and trip_distance < best_trip_distance):
                    matched_rider = rider
                    max_reward = reward
                    best_trip_distance = trip_distance

        return matched_rider if max_reward > 0 else None


    def find_driver_for_rider(self, rider, rewards, distance_matrix):
        """
        Find the most compatible driver for a rider based on compatibility sets and rewards.
        Searches across all drivers in the system.

        :param rider: The arriving rider.
        :param rewards: The reward matrix.
        :param distance_matrix: The shortest path distance matrix between all nodes.
        :return: The matched driver, or None if no match is found or reward is ≤ 0.
        """
        if self.print:
            print(f"Current active drivers: {[d.arrival_time for d in self.active_drivers]}")

        if not self.active_drivers:
            return None

        matched_driver = None
        max_reward = -float('inf')  # Start with the lowest possible reward
        best_trip_distance = float('inf')  # For tie-breaking based on trip distance

        for driver in self.active_drivers:
            # Check if the rider's location is in the driver's compatibility set
            if rider.location in driver.compatibility_set:
                # Calculate reward and trip distance
                reward = rewards[driver.type][rider.type]
                trip_distance = abs(driver.location - rider.location)  # Fallback

                # Skip if reward is non-positive
                if reward <= 0:
                    continue

                # Update the matched driver if this one is better
                if reward > max_reward or (reward == max_reward and trip_distance < best_trip_distance):
                    matched_driver = driver
                    max_reward = reward
                    best_trip_distance = trip_distance

        return matched_driver if max_reward > 0 else None


    
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
        print(f"Average Wait Time: {average_wait_time:.2f} units\n")
        
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

  