import numpy as np
import heapq
import networkx as nx

# Basic data structures

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

# City layout generation

def generate_simple_grid(num_nodes):
    grid_size = int(np.ceil(np.sqrt(num_nodes)))
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        if i % grid_size != grid_size - 1 and i + 1 < num_nodes:
            adjacency_matrix[i][i + 1] = adjacency_matrix[i + 1][i] = 1
        if i + grid_size < num_nodes:
            adjacency_matrix[i][i + grid_size] = adjacency_matrix[i + grid_size][i] = 1

    return adjacency_matrix

# Event generation with heterogeneous rates

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time):
    current_time = 0
    rider_count = 0
    driver_count = 0

    while current_time < simulation_time:
        # Generate rider arrivals
        next_rider_time = current_time + np.random.exponential(1/rate_riders)
        if next_rider_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate_riders)
            rider_location = np.random.randint(0, num_nodes)
            rider = Rider(next_rider_time, rider_location, patience, sojourn_time)
            event_queue.add_event(Event(next_rider_time, 'rider_arrival', rider))
            rider_count += 1

        # Generate driver arrivals
        next_driver_time = current_time + np.random.exponential(1/rate_drivers)
        if next_driver_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate_drivers)
            driver_location = np.random.randint(0, num_nodes)
            driver = Driver(next_driver_time, driver_location, patience, sojourn_time)
            event_queue.add_event(Event(next_driver_time, 'driver_arrival', driver))
            driver_count += 1

        current_time = min(next_rider_time, next_driver_time)

    print(f"Total riders generated: {rider_count}")
    print(f"Total drivers generated: {driver_count}")

# Realization Graph

class RealizationGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_entity(self, entity):
        self.graph.add_node(entity)

    def remove_entity(self, entity):
        if entity in self.graph:
            self.graph.remove_node(entity)

    def add_match(self, rider, driver):
        self.graph.add_edge(rider, driver)

    def display_graph(self):
        print("Nodes:", self.graph.nodes)
        print("Edges:", self.graph.edges)

# Simulation runner with graph

def run_simulation(matching_algorithm, num_nodes, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, simulation_time, num_timesteps):
    # Generate city layout
    adjacency_matrix = generate_simple_grid(num_nodes)

    # Initialize event queue
    event_queue = EventQueue()

    # Generate events
    generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)

    # Initialize realization graph
    realization_graph = RealizationGraph()

    # Run simulation
    current_time = 0
    available_riders = []
    available_drivers = []
    matched_pairs = []

    while True:
        event = event_queue.get_next_event()
        if event is None or event.time > simulation_time:
            break

        # Align current_time to the nearest multiple of num_timesteps
        current_time = event.time - (event.time % num_timesteps)

        if event.event_type == 'rider_arrival':
            available_riders.append(event.entity)
            realization_graph.add_entity(event.entity)
        elif event.event_type == 'driver_arrival':
            available_drivers.append(event.entity)
            realization_graph.add_entity(event.entity)

        # Call the matching algorithm
        new_matches = matching_algorithm(available_riders, available_drivers, adjacency_matrix, current_time, num_timesteps)
        matched_pairs.extend(new_matches)

        # Update the graph with new matches
        for rider, driver, _ in new_matches:
            if rider in available_riders:
                available_riders.remove(rider)
            if driver in available_drivers:
                available_drivers.remove(driver)
            realization_graph.add_match(rider, driver)
            realization_graph.remove_entity(rider)
            realization_graph.remove_entity(driver)

        # Remove expired entities
        available_riders = [r for r in available_riders if current_time - r.arrival_time < r.patience]
        available_drivers = [d for d in available_drivers if current_time - d.arrival_time < d.patience]

    print(f"Total matched pairs: {len(matched_pairs)}")
    print(f"Remaining available riders: {len(available_riders)}")
    print(f"Remaining available drivers: {len(available_drivers)}")

    return matched_pairs, available_riders, available_drivers

# Updated matching algorithm
def match_every_n_timesteps(available_riders, available_drivers, adjacency_matrix, current_time, num_timesteps=4):
    # Only match if current_time is a multiple of num_timesteps
    if current_time % num_timesteps != 0:
        return []
    
    matches = []
    
    # Match riders and drivers based on their availability
    while available_riders and available_drivers:
        rider = available_riders[0]  # Take the first available rider
        driver = available_drivers[0]  # Take the first available driver
        
        # Add the pair to the matches list
        matches.append((rider, driver, current_time))
        
        # Remove them from the available lists
        available_riders.remove(rider)
        available_drivers.remove(driver)

    return matches

# Example usage
if __name__ == "__main__":
    # Simulation parameters
    num_nodes = 10
    rate_riders = 10
    rate_drivers = 10
    sojourn_rate_riders = 0.4
    sojourn_rate_drivers = 0.5
    simulation_time = 100
    num_timesteps = 4

    # Run simulation
    matched_pairs, unmatched_riders, unmatched_drivers = run_simulation(
        match_every_n_timesteps, num_nodes, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, simulation_time, num_timesteps
    )

    # Print basic results
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Unmatched riders: {len(unmatched_riders)}")
    print(f"Unmatched drivers: {len(unmatched_drivers)}")
