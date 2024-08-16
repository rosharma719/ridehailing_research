import numpy as np
import heapq

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

# Event generation

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate, num_nodes, simulation_time):
    current_time = 0
    while current_time < simulation_time:
        # Generate rider arrivals
        next_rider_time = current_time + np.random.exponential(1/rate_riders)
        if next_rider_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate)
            rider_location = np.random.randint(0, num_nodes)
            rider = Rider(next_rider_time, rider_location, patience, sojourn_time)
            event_queue.add_event(Event(next_rider_time, 'rider_arrival', rider))

        # Generate driver arrivals
        next_driver_time = current_time + np.random.exponential(1/rate_drivers)
        if next_driver_time < simulation_time:
            patience = np.random.randint(1, 10)
            sojourn_time = np.random.exponential(1 / sojourn_rate)
            driver_location = np.random.randint(0, num_nodes)
            driver = Driver(next_driver_time, driver_location, patience, sojourn_time)
            event_queue.add_event(Event(next_driver_time, 'driver_arrival', driver))

        current_time = min(next_rider_time, next_driver_time)

# Simulation runner

def run_simulation(matching_algorithm, num_nodes, rate_riders, rate_drivers, sojourn_rate, simulation_time):
    # Generate city layout
    adjacency_matrix = generate_simple_grid(num_nodes)

    # Initialize event queue
    event_queue = EventQueue()

    # Generate events
    generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate, num_nodes, simulation_time)

    # Run simulation
    current_time = 0
    available_riders = []
    available_drivers = []
    matched_pairs = []

    while True:
        event = event_queue.get_next_event()
        if event is None or event.time > simulation_time:
            break

        current_time = event.time

        if event.event_type == 'rider_arrival':
            available_riders.append(event.entity)
        elif event.event_type == 'driver_arrival':
            available_drivers.append(event.entity)

        # Call the matching algorithm
        new_matches = matching_algorithm(available_riders, available_drivers, adjacency_matrix, current_time)
        matched_pairs.extend(new_matches)

        # Remove matched entities
        for rider, driver, _ in new_matches:
            available_riders.remove(rider)
            available_drivers.remove(driver)

        # Remove expired entities
        available_riders = [r for r in available_riders if current_time - r.arrival_time < r.patience]
        available_drivers = [d for d in available_drivers if current_time - d.arrival_time < d.patience]

    return matched_pairs, available_riders, available_drivers

# Placeholder for matching algorithm (to be implemented by the user)
def dummy_matching_algorithm(available_riders, available_drivers, adjacency_matrix, current_time):
    # This is a placeholder. The actual matching algorithm should be implemented by the user.
    return []

# Example usage
if __name__ == "__main__":
    # Simulation parameters
    num_nodes = 10
    rate_riders = 5
    rate_drivers = 5
    sojourn_rate = 0.4
    simulation_time = 100

    # Run simulation
    matched_pairs, unmatched_riders, unmatched_drivers = run_simulation(
        dummy_matching_algorithm, num_nodes, rate_riders, rate_drivers, sojourn_rate, simulation_time
    )

    # Print basic results
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Unmatched riders: {len(unmatched_riders)}")
    print(f"Unmatched drivers: {len(unmatched_drivers)}")