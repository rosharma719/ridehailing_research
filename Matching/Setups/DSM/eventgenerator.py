import heapq
import numpy as np
import random

# Rider and Driver Classes
class Rider:
    def __init__(self, arrival_time, location, patience, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time
        self.abandonment_time = arrival_time + patience  # Time when the rider will abandon if unmatched
        self.type = f"Rider {location}"  # Includes location in the type
        self.label = None  # Will be set to 'active' or 'passive'

class Driver:
    def __init__(self, arrival_time, location, patience, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.patience = patience
        self.sojourn_time = sojourn_time
        self.abandonment_time = arrival_time + patience  # Time when the driver will abandon if unmatched
        self.type = f"Driver {location}"  # Includes location in the type
        self.label = None  # Will be set to 'active' or 'passive'

# Event Class with Detailed Event Types
class Event:
    def __init__(self, time, event_type, entity=None):
        self.time = time
        self.event_type = event_type
        self.entity = entity

    def __lt__(self, other):
        return self.time < other.time

# Event Queue to Manage the Timeline
class EventQueue:
    def __init__(self):
        self.queue = []

    def add_event(self, event):
        heapq.heappush(self.queue, event)

    def get_next_event(self):
        return heapq.heappop(self.queue) if self.queue else None

    def is_empty(self):
        return len(self.queue) == 0

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time):
    current_time = 0

    while current_time < simulation_time:
        # Generate the next rider arrival time
        next_rider_time = current_time + np.random.exponential(1 / rate_riders)
        if next_rider_time < simulation_time:
            patience = np.random.uniform(1, 10)  # Random patience time
            sojourn_time = np.random.exponential(1 / sojourn_rate_riders)
            rider_location = np.random.randint(0, num_nodes)
            
            # Ensure the destination is different from the rider's location
            rider_destination = rider_location
            while rider_destination == rider_location:
                rider_destination = np.random.randint(0, num_nodes)
            
            rider = Rider(next_rider_time, rider_location, patience, sojourn_time)
            event_queue.add_event(Event(next_rider_time, 'arrival', rider))
            event_queue.add_event(Event(rider.abandonment_time, 'abandonment', rider))

        # Generate the next driver arrival time
        next_driver_time = current_time + np.random.exponential(1 / rate_drivers)
        if next_driver_time < simulation_time:
            patience = np.random.uniform(1, 10)  # Random patience time
            sojourn_time = np.random.exponential(1 / sojourn_rate_drivers)
            driver_location = np.random.randint(0, num_nodes)
            driver = Driver(next_driver_time, driver_location, patience, sojourn_time)
            event_queue.add_event(Event(next_driver_time, 'arrival', driver))
            event_queue.add_event(Event(driver.abandonment_time, 'abandonment', driver))

        # Move forward to the next event time
        current_time = min(next_rider_time, next_driver_time)
