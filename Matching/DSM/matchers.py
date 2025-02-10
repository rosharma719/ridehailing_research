from eventgenerator import *
from utils import *

from scipy.sparse.csgraph import shortest_path

def greedy_matcher(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix):
    print_stuff = False
    """
    Greedy matcher that pairs riders with drivers as soon as they arrive,
    selecting the highest-reward rider across all locations.

    :param event_queue: The queue of events (arrival, abandonment).
    :param rewards: The reward matrix for matches.
    :param results: Results from QB optimization.
    :param lambda_i: Arrival rates for active types.
    :param lambda_j: Arrival rates for passive types.
    :param mu_i: Abandonment rates for active types.
    :param adjacency_matrix: Adjacency matrix representing the graph.
    """

    # Compute the shortest path distance matrix
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)

    realization_graph = RealizationGraph()

    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        if print_stuff: 
            print(f"\nProcessing event: {event.event_type} for entity: {event.entity}")

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                if print_stuff: 
                    print(f"Driver arrived at location {event.entity.location} at time {event.entity.arrival_time}.")
                realization_graph.add_driver(event.entity)

                # Get all waiting riders
                waiting_riders = realization_graph.passive_riders  

                if waiting_riders:
                    best_rider = None
                    best_reward = -float('inf')
                    best_trip_distance = float('inf')

                    if print_stuff:
                        print("\n=== Possible Matches for Driver ===")
                    for rider in waiting_riders:
                        if rider.location in event.entity.compatibility_set:
                            reward = rewards[event.entity.type][rider.type]
                            trip_distance = distance_matrix[event.entity.location][rider.location]
                            
                            if print_stuff: 
                                print(f"Rider at Node {rider.location}: Reward {reward}, Distance {trip_distance}")

                            # Update best match if:
                            # 1. This rider has a higher reward
                            # 2. If same reward, this rider has a shorter trip distance
                            if reward > best_reward or (reward == best_reward and trip_distance < best_trip_distance):
                                best_rider = rider
                                best_reward = reward
                                best_trip_distance = trip_distance

                    if best_rider:
                        if print_stuff: 
                            print(f"\n>> Selected Best Match: Rider at {best_rider.location}, "
                                f"Reward: {best_reward}, Distance: {best_trip_distance}")

                        perform_match(
                            realization_graph,
                            driver=event.entity,
                            rider=best_rider,
                            rewards=rewards,
                            distance_matrix=distance_matrix,
                            event_queue=event_queue,
                        )

                        if print_stuff:
                            print(f"Driver at {event.entity.location} matched with Rider at {best_rider.location}.")

            elif isinstance(event.entity, Rider):
                if print_stuff: 
                    print(f"Rider arrived at location {event.entity.location} at time {event.entity.arrival_time}.")
                realization_graph.add_rider(event.entity)

                # Find the closest compatible driver
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards, distance_matrix)

                if matched_driver:
                    if print_stuff:
                        print(f"\n>> Rider at {event.entity.location} matched with Driver at {matched_driver.location}.")
                    perform_match(
                        realization_graph,
                        driver=matched_driver,
                        rider=event.entity,
                        rewards=rewards,
                        distance_matrix=distance_matrix,
                        event_queue=event_queue,
                    )

        elif event.event_type == 'abandonment':
            # Handle abandonment with precise state tracking
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    if print_stuff:
                        print(f"Driver at {event.entity.location} abandoned at time {event.entity.abandonment_time}.")
                    realization_graph.remove_driver(event.entity)
                else:
                    raise RuntimeError(f"Driver abandonment inconsistency: {event.entity} not in active_drivers.")

            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    if print_stuff:
                        print(f"Rider at {event.entity.location} abandoned at time {event.entity.abandonment_time}.")
                    realization_graph.remove_rider(event.entity)
                else:
                    raise RuntimeError(f"Rider abandonment inconsistency: {event.entity} not in passive_riders.")

    print("Event processing completed.")
    realization_graph.print_summary()


def greedy_auto_label(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix):
    

    # Compute the shortest path distance matrix
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)

    realization_graph = RealizationGraph()
    print_stuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Generate label and compatibility sets for driver
                driver_label, label_to_set_map = generate_label(
                    False,
                    event.entity.location,
                    results,
                    lambda_i,
                    lambda_j,
                    mu_i
                )
                event.entity.label = driver_label  # Assign label to the driver
                event.entity.compatibility_set = label_to_set_map.get(driver_label, [])  # Assign compatibility set
                realization_graph.add_driver(event.entity)

                if print_stuff:
                    print(f"Driver labeled as {driver_label} at location {event.entity.location}.")

                # Check for immediate matches with waiting riders
                waiting_riders = realization_graph.get_waiting_riders_at_node(event.entity.location)
                if waiting_riders:
                    matched_rider = waiting_riders[0]
                    if print_stuff:
                        print(f"Driver at {event.entity.location} attempting to match with Rider at {matched_rider.location}.")
                    perform_match(
                        realization_graph,
                        driver=event.entity,
                        rider=matched_rider,
                        rewards=rewards,
                        distance_matrix=distance_matrix,
                        event_queue=event_queue,
                    )

            elif isinstance(event.entity, Rider):
                # Riders are trivially passive; assign a fixed label
                event.entity.label = 0
                realization_graph.add_rider(event.entity)

                if print_stuff:
                    print(f"Rider added at location {event.entity.location}.")

                # Find the closest compatible driver
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards, distance_matrix)

                if matched_driver:
                    if print_stuff:
                        print(f"Rider at {event.entity.location} attempting to match with Driver at {matched_driver.location}.")
                    perform_match(
                        realization_graph,
                        driver=matched_driver,
                        rider=event.entity,
                        rewards=rewards,
                        distance_matrix=distance_matrix,
                        event_queue=event_queue,
                    )

        elif event.event_type == 'abandonment':
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    realization_graph.remove_driver(event.entity)
                    if print_stuff:
                        print(f"Driver at {event.entity.location} abandoned at time {event.entity.abandonment_time}.")
                else:
                    raise RuntimeError(f"Driver abandonment inconsistency: {event.entity} not in active_drivers.")

            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    realization_graph.remove_rider(event.entity)
                    if print_stuff:
                        print(f"Rider at {event.entity.location} abandoned at time {event.entity.abandonment_time}.")
                else:
                    raise RuntimeError(f"Rider abandonment inconsistency: {event.entity} not in passive_riders.")

    print("Event processing completed.")
    realization_graph.print_summary()


def greedy_auto_label_nonperish(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix):
    """
    Process events and generate labels for drivers and riders using the flow matrix \tilde{x}_{i,j}.
    Riders wait at their node until a driver arrives, matching to the first available driver at the same node.
    """
    # Compute the shortest path distance matrix
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)

    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Generate label and compatibility sets for driver
                driver_label, label_to_set_map = generate_label(
                    False,
                    event.entity.location,
                    results,
                    lambda_i,
                    lambda_j,
                    mu_i
                )
                event.entity.label = driver_label  # Assign label to the driver
                event.entity.compatibility_set = label_to_set_map.get(driver_label, [])  # Assign compatibility set
                realization_graph.add_driver(event.entity)

                # Check if there's a waiting rider at the driver's node
                waiting_riders = realization_graph.get_waiting_riders_at_node(event.entity.location)
                if waiting_riders:
                    matched_rider = waiting_riders[0]  # Match the first waiting rider
                    realization_graph.remove_driver(event.entity)
                    realization_graph.remove_rider(matched_rider)

                    # Calculate match stats (wait time, distance, reward)
                    wait_time = event.entity.arrival_time - matched_rider.arrival_time
                    trip_distance = distance_matrix[event.entity.location, matched_rider.location]  # Use distance matrix
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


def greedy_auto_label_nonperish_floor(event_queue, rewards, results, lambda_i, lambda_j, mu_i, thickness_floor, adjacency_matrix):
    """
    Process events and generate labels for drivers and riders using the flow matrix.
    Riders wait at their node until a driver arrives, matching to the first available driver
    only if there are more than the thickness_floor drivers at the node.
    """
    # Compute the shortest path distance matrix
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)

    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Generate label and compatibility sets for driver
                driver_label, label_to_set_map = generate_label(
                    False,
                    event.entity.location,
                    results,
                    lambda_i,
                    lambda_j,
                    mu_i
                )
                event.entity.label = driver_label  # Assign label to the driver
                event.entity.compatibility_set = label_to_set_map.get(driver_label, [])  # Assign compatibility set
                realization_graph.add_driver(event.entity)

                # Check if there's a waiting rider at the driver's node
                waiting_riders = realization_graph.get_waiting_riders_at_node(event.entity.location)
                waiting_drivers = realization_graph.get_waiting_drivers_at_node(event.entity.location)

                # Only match if the number of waiting drivers exceeds the threshold
                if len(waiting_drivers) > thickness_floor and waiting_riders:
                    matched_rider = waiting_riders[0]  # Match the first waiting rider
                    realization_graph.remove_driver(event.entity)
                    realization_graph.remove_rider(matched_rider)

                    # Calculate match stats (wait time, distance, reward)
                    wait_time = event.entity.arrival_time - matched_rider.arrival_time
                    trip_distance = distance_matrix[event.entity.location, matched_rider.location]
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
                    print(f"Driver at Node {event.entity.location} waiting. Not enough drivers (threshold: {thickness_floor}).")

            elif isinstance(event.entity, Rider):
                # Riders are trivially passive, so no need to call generate_label
                event.entity.label = 0
                realization_graph.add_rider(event.entity)

        elif event.event_type == 'abandonment':
            # Handle abandonment of drivers or riders
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    realization_graph.remove_driver(event.entity)
            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    realization_graph.remove_rider(event.entity)

    print("Event processing completed.")
    realization_graph.print_summary()

