import heapq
import numpy as np
import random

class Rider:
    def __init__(self, arrival_time, location, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.sojourn_time = sojourn_time  # Time the rider is willing to wait
        self.abandonment_time = arrival_time + sojourn_time  # Time when the rider will abandon if unmatched
        self.type = f"Passive Rider Node {location}"  # Match the type format to the rewards dictionary
        self.label = None  # Will be set to 'active' or 'passive'

class Driver:
    def __init__(self, arrival_time, location, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.sojourn_time = sojourn_time  # Time the driver is willing to wait
        self.abandonment_time = arrival_time + sojourn_time  # Time when the driver will abandon if unmatched
        self.type = f"Active Driver Node {location}"  # Match the type format to the rewards dictionary
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
    
    def clone(self):
        new_queue = EventQueue()
        new_queue.queue = list(self.queue)  # Copy the heap
        heapq.heapify(new_queue.queue)  # Re-heapify the copied queue
        return new_queue

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time):
    """
    Generate events (arrival and abandonment) for each node separately to support scaling by node.
    
    :param event_queue: The event queue where events are stored.
    :param rate_riders: The arrival rate of riders (global rate).
    :param rate_drivers: The arrival rate of drivers (global rate).
    :param sojourn_rate_riders: The sojourn time rate for riders.
    :param sojourn_rate_drivers: The sojourn time rate for drivers.
    :param num_nodes: Number of nodes in the system.
    :param simulation_time: Total simulation time.
    """
    
    current_time = 0

    # Generate arrivals for each node separately
    for node in range(num_nodes):
        node_time = 0  # Track the time for this node
        
        while node_time < simulation_time:
            # Generate the next rider arrival time for this node
            next_rider_time = node_time + np.random.exponential(1 / rate_riders)
            if next_rider_time < simulation_time:
                sojourn_time = np.random.exponential(1 / sojourn_rate_riders)

                rider = Rider(next_rider_time, node, sojourn_time)
                event_queue.add_event(Event(next_rider_time, 'arrival', rider))

                # Add the abandonment event if it happens within the simulation time
                abandonment_time = rider.abandonment_time
                if abandonment_time < simulation_time:
                    event_queue.add_event(Event(abandonment_time, 'abandonment', rider))

            # Generate the next driver arrival time for this node
            next_driver_time = node_time + np.random.exponential(1 / rate_drivers)
            if next_driver_time < simulation_time:
                sojourn_time = np.random.exponential(1 / sojourn_rate_drivers)
                driver = Driver(next_driver_time, node, sojourn_time)
                event_queue.add_event(Event(next_driver_time, 'arrival', driver))

                # Add the abandonment event if it happens within the simulation time
                abandonment_time = driver.abandonment_time
                if abandonment_time < simulation_time:
                    event_queue.add_event(Event(abandonment_time, 'abandonment', driver))

            # Move forward to the next event time for this node
            node_time = min(next_rider_time, next_driver_time)
            node_time = min(node_time, simulation_time)


def remove_abandonment_event(event_queue, entity):
    """
    Remove the abandonment event of the entity (driver or rider) from the event queue if they are matched.
    
    :param event_queue: The event queue where events are stored
    :param entity: The entity (driver or rider) whose abandonment event needs to be removed
    """
    updated_queue = []

    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        # Check if the event is not the abandonment event for the matched entity
        if event.event_type == 'abandonment' and event.entity == entity:
            continue  # Skip this abandonment event
        updated_queue.append(event)

    # Rebuild the event queue without the matched entity's abandonment event
    for event in updated_queue:
        event_queue.add_event(event)
