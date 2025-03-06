import heapq
import numpy as np
import random


def remove_abandonment_event(event_queue, entity):
    """
    Remove the abandonment event of the entity (driver or rider) from the event queue if they are matched.
    """
    updated_queue = []
    
    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        if event.event_type == 'abandonment' and event.entity == entity:
            continue
        updated_queue.append(event)

    # Rebuild the event queue
    for event in updated_queue:
        event_queue.add_event(event)


class Rider:
    def __init__(self, arrival_time, location, sojourn_time):
        self.arrival_time = arrival_time
        self.location = location
        self.sojourn_time = sojourn_time  # Time the rider is willing to wait
        self.abandonment_time = arrival_time + sojourn_time  # Time when the rider will abandon if unmatched
        self.type = f"Passive Rider Node {location}"  # Match the type format to the rewards dictionary
        self.label = None  # Will be set to 'active' or 'passive'
    
    def __hash__(self):
        return hash((round(self.arrival_time, 6), self.location, round(self.sojourn_time, 6)))

    def __eq__(self, other):
        return (round(self.arrival_time, 6), self.location, round(self.sojourn_time, 6)) == \
               (round(other.arrival_time, 6), other.location, round(other.sojourn_time, 6))



class Driver:
    def __init__(self, arrival_time, location, sojourn_time, num_nodes):
        self.arrival_time = arrival_time
        self.location = location
        self.sojourn_time = sojourn_time  # Time the driver is willing to wait
        self.abandonment_time = arrival_time + sojourn_time  # Time when the driver will abandon if unmatched
        self.type = f"Active Driver Node {location}"  # Match the type format to the rewards dictionary
        self.label = None  # Will be set to 'active' or 'passive'
        self.compatibility_set = list(range(num_nodes))  # Compatible with all nodes

    def __hash__(self):
        return hash((round(self.arrival_time, 6), self.location, round(self.sojourn_time, 6)))

    def __eq__(self, other):
        return (round(self.arrival_time, 6), self.location, round(self.sojourn_time, 6)) == \
               (round(other.arrival_time, 6), other.location, round(other.sojourn_time, 6))
    


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
    
    def print_queue(self):
            """
            Print the details of all events in the queue in a human-readable format.
            """
            print("Current Event Queue:")
            for event in sorted(self.queue, key=lambda e: e.time):  # Sort by event time for readability
                entity_type = type(event.entity).__name__ if event.entity else "None"
                entity_details = f"Type: {entity_type}, Location: {getattr(event.entity, 'location', 'N/A')}, Arrival: {getattr(event.entity, 'arrival_time', 'N/A')}" if event.entity else "None"
                print(f"Time: {event.time:.6f}, Event Type: {event.event_type}, Entity: [{entity_details}]")
            print("\n")

def generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time):
    """
    Generate events (arrival and abandonment) for each node separately.
    Ensures every arrival has a corresponding abandonment event tracked.
    """
    current_time = 0
    count_missed = 0  # Count abandonments after simulation end
    expected_abandonments = 0  # Count all potential abandonments
    total_arrivals = 0  # Track total arrivals
    
    for node in range(num_nodes):
        node_time = 0  # Track the time for this node
        while node_time < simulation_time:
            # Generate the next rider arrival time
            next_rider_time = round(node_time + np.random.exponential(1 / rate_riders), 6)
            if next_rider_time < simulation_time:
                sojourn_time = round(np.random.exponential(1 / sojourn_rate_riders), 6)
                rider = Rider(next_rider_time, node, sojourn_time)
                event_queue.add_event(Event(next_rider_time, 'arrival', rider))
                total_arrivals += 1  # Count every arrival
                
                # Always track abandonment events, whether within simulation time or not
                abandonment_time = rider.abandonment_time
                expected_abandonments += 1  # Count every potential abandonment
                
                if abandonment_time < simulation_time:
                    event_queue.add_event(Event(abandonment_time, 'abandonment', rider))
                else:
                    count_missed += 1  # Count abandonments after simulation end
            
            # Generate the next driver arrival time
            next_driver_time = round(node_time + np.random.exponential(1 / rate_drivers), 6)
            if next_driver_time < simulation_time:
                sojourn_time = round(np.random.exponential(1 / sojourn_rate_drivers), 6)
                driver = Driver(next_driver_time, node, sojourn_time, num_nodes)
                event_queue.add_event(Event(next_driver_time, 'arrival', driver))
                total_arrivals += 1  # Count every arrival
                
                # Always track abandonment events, whether within simulation time or not
                abandonment_time = driver.abandonment_time
                expected_abandonments += 1  # Count every potential abandonment
                
                if abandonment_time < simulation_time:
                    event_queue.add_event(Event(abandonment_time, 'abandonment', driver))
                else:
                    count_missed += 1  # Count abandonments after simulation end
            
            node_time = min(next_rider_time, next_driver_time)
            node_time = min(node_time, simulation_time)
    
    print(f"\nTotal Arrivals: {total_arrivals}")
    print(f"Total Abandonments: {expected_abandonments}")
    print(f"Abandonments Within Timeline: {expected_abandonments - count_missed}")
    print(f"Abandonments Outside Timeline: {count_missed}")
    
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
