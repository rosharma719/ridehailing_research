from eventgenerator import * 
from utils import *

def greedy_auto_label_floor(event_queue, rewards, results, lambda_i, lambda_j, mu_i):
    """
    Process events and generate labels for drivers and riders using the flow matrix \tilde{x}_{i,j}.
    Riders wait at their node until a driver arrives, matching to the first available driver at the same node.
    """
    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Generate label for driver based on the results
                driver_label = generate_label(
                    False,
                    event.entity.location,
                    results,
                    lambda_i,
                    lambda_j,
                    mu_i
                )
                event.entity.label = driver_label  # Assign label to the driver
                realization_graph.add_driver(event.entity)

                # Check if there's a waiting rider at the driver's node
                waiting_riders = realization_graph.get_waiting_riders_at_node(event.entity.location)
                if waiting_riders:
                    matched_rider = waiting_riders[0]  # Match the first waiting rider
                    realization_graph.remove_driver(event.entity)
                    realization_graph.remove_rider(matched_rider)

                    # Calculate match stats (wait time, distance, reward)
                    wait_time = event.entity.arrival_time - matched_rider.arrival_time
                    trip_distance = abs(event.entity.location - matched_rider.location)
                    reward = rewards[event.entity.type][matched_rider.type]

                    realization_graph.total_wait_time += wait_time
                    realization_graph.total_trip_distance += trip_distance
                    realization_graph.total_rewards += reward
                    realization_graph.num_drivers_matched += 1
                    realization_graph.num_riders_matched += 1

                    # Remove their abandonment events from the event queue
                    remove_abandonment_event(event_queue, matched_rider)
                    remove_abandonment_event(event_queue, event.entity)

                    if printStuff:
                        print(f"Matched Driver at Node {event.entity.location} with Rider at Node {matched_rider.location}")
                elif printStuff:
                    print(f"No riders waiting at Node {event.entity.location}. Driver is waiting.")

            elif isinstance(event.entity, Rider):
                # Riders are trivially passive, so no need to call generate_label
                event.entity.label = 0
                realization_graph.add_rider(event.entity)

                # Riders don't perish instantly—drivers arriving later may match them

        elif event.event_type == 'abandonment':
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    realization_graph.remove_driver(event.entity)
                    if printStuff:
                        print(f"Driver abandoned at Node {event.entity.location}")
            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    realization_graph.remove_rider(event.entity)
                    if printStuff:
                        print(f"Rider abandoned at Node {event.entity.location}")

    print("Event processing completed.")
    realization_graph.print_summary()


def greedy_auto_label(event_queue, rewards, results, lambda_i, lambda_j, mu_i):
    """
    Process events and generate labels for drivers and riders using the flow matrix \tilde{x}_{i,j}.
    
    :param event_queue: The queue of events (arrival, abandonment)
    :param rewards: The reward matrix for matches
    :param results: The flow rates, abandonment rates, and unmatched rates from QB optimization
    :param lambda_i: Arrival rates for active types
    :param lambda_j: Arrival rates for passive types
    :param mu_i: Sojourn rates for active types
    """
    
    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Generate label for driver based on the results
                driver_label = generate_label(
                    False, 
                    event.entity.location, 
                    results, 
                    lambda_i, 
                    lambda_j, 
                    mu_i
                )
                event.entity.label = driver_label  # Assign label to the driver
                realization_graph.add_driver(event.entity)

            elif isinstance(event.entity, Rider):
                # Riders are trivially passive, so no need to call generate_label
                event.entity.label = 0
                realization_graph.add_rider(event.entity)

                # Try to find a match for the rider
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards)

                if matched_driver:
                    # Remove the matched rider and driver from the system
                    realization_graph.remove_rider(event.entity)

                    # Remove their abandonment events from the event queue
                    remove_abandonment_event(event_queue, matched_driver)

                    if printStuff:
                        print(f"Matched {event.entity.type} at {event.entity.location} with {matched_driver.type}")
                elif printStuff:
                    print(f"No available drivers for {event.entity.type} at {event.entity.location}. Rider abandoned.")

        elif event.event_type == 'abandonment':
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    realization_graph.remove_driver(event.entity)
                    if printStuff:
                        print(f"{event.entity.type} at {event.entity.location} left the system")
            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    realization_graph.remove_rider(event.entity)
                    if printStuff:
                        print(f"{event.entity.type} at {event.entity.location} left the system")

    print("Event processing completed.")
    realization_graph.print_summary()

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
