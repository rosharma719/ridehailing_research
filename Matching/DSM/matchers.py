import logging
from eventgenerator import *
from utils import *

from scipy.sparse.csgraph import shortest_path

def greedy_matcher(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix):
    logging.info("Starting greedy_matcher")

    rider_thickness_values = []
    driver_thickness_values = []

    
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)
    realization_graph = RealizationGraph()
    abandonments = 0 

    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        logging.info(f"Processing event: {event.event_type} | Entity: {event.entity} | Arrival Time: {event.entity.arrival_time}")

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                rider_thickness_values.append(len(realization_graph.passive_riders))


                logging.info(f"Driver {event.entity.arrival_time} arrived | Location: {event.entity.location}")
                realization_graph.add_driver(event.entity)

                matched_rider = realization_graph.find_rider_for_driver(event.entity, rewards, distance_matrix)

                if matched_rider:
                    logging.info(f"Matched Driver {event.entity.arrival_time} | Location: {event.entity.location} with Rider {matched_rider.arrival_time} | Location: {matched_rider.location}")
                    perform_match(realization_graph, driver=event.entity, rider=matched_rider, rewards=rewards, distance_matrix=distance_matrix, event_queue=event_queue)

            elif isinstance(event.entity, Rider):
                driver_thickness_values.append(len(realization_graph.active_drivers))


                logging.info(f"Rider {event.entity.arrival_time} arrived | Location: {event.entity.location}")
                realization_graph.add_rider(event.entity)

                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards, distance_matrix)

                if matched_driver:
                    logging.info(f"Matched Rider {event.entity.arrival_time} | Location: {event.entity.location} with Driver {matched_driver.arrival_time} | Location: {matched_driver.location}")
                    perform_match(realization_graph, driver=matched_driver, rider=event.entity, rewards=rewards, distance_matrix=distance_matrix, event_queue=event_queue)

        elif event.event_type == 'abandonment':
            abandonments += 1
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    logging.info(f"Driver {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_driver(event.entity)
                else:
                    raise RuntimeError(f"Driver abandonment inconsistency: {event.entity} not in active_drivers.")
            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    logging.info(f"Rider {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_rider(event.entity)
                else:
                    raise RuntimeError(f"Rider abandonment inconsistency: {event.entity} not in passive_riders.")

    logging.info("Event processing completed.")
    print(abandonments, "abandonments")

    print(f"Average rider-side thickness (drivers available at rider arrival): {mean(driver_thickness_values):.3f}")
    print(f"Average driver-side thickness (riders available at driver arrival): {mean(rider_thickness_values):.3f}")

    realization_graph.print_summary()

def greedy_auto_label(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix, explicitly_remove_riders):
    logging.info("Starting greedy_auto_label")
    
    rider_thickness_values = []
    driver_thickness_values = []


    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)
    realization_graph = RealizationGraph()
    abandonments = 0

    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        logging.info(f"Processing event: {event.event_type} | Entity: {event.entity} | Arrival Time: {event.entity.arrival_time}")

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                rider_thickness_values.append(len(realization_graph.passive_riders))

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
                
                logging.info(f"Driver {event.entity.arrival_time} arrived | Location: {event.entity.location}")
                realization_graph.add_driver(event.entity)

            elif isinstance(event.entity, Rider):
                driver_thickness_values.append(len(realization_graph.active_drivers))


                # Riders are trivially passive; assign a fixed label
                event.entity.label = 0
                
                logging.info(f"Rider {event.entity.arrival_time} arrived | Location: {event.entity.location}")
                realization_graph.add_rider(event.entity)

                # Find the closest compatible driver
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards, distance_matrix)

                if matched_driver:
                    logging.info(f"Matched Rider {event.entity.arrival_time} | Location: {event.entity.location} with Driver {matched_driver.arrival_time} | Location: {matched_driver.location}")
                    perform_match(
                        realization_graph,
                        driver=matched_driver,
                        rider=event.entity,
                        rewards=rewards,
                        distance_matrix=distance_matrix,
                        event_queue=event_queue,
                    )
                elif explicitly_remove_riders: 
                    realization_graph.remove_rider(event.entity)
                    remove_abandonment_event(event_queue, event.entity)
                    abandonments+=1
                    logging.info(f"Rider {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")


        elif event.event_type == 'abandonment':
            abandonments += 1
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    logging.info(f"Driver {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_driver(event.entity)
                else:
                    raise RuntimeError(f"Driver abandonment inconsistency: {event.entity} not in active_drivers.")

            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    logging.info(f"Rider {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_rider(event.entity)
                else:
                    raise RuntimeError(f"Rider abandonment inconsistency: {event.entity} not in passive_riders.")

    logging.info("Event processing completed.")
    print(abandonments, "abandonments")

    print(f"Average rider-side thickness (drivers available at rider arrival): {mean(driver_thickness_values):.3f}")
    print(f"Average driver-side thickness (riders available at driver arrival): {mean(rider_thickness_values):.3f}")

    realization_graph.print_summary()

def nonperish_bidirectional_GAL(event_queue, rewards, results, lambda_i, lambda_j, mu_i, adjacency_matrix):
    logging.info("Starting nonperish_bidirectional_GAL")
    abandonments = 0

    rider_thickness_values = []
    driver_thickness_values = []

    
    # Compute the shortest path distance matrix
    distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=True)
    realization_graph = RealizationGraph()

    while not event_queue.is_empty():
        event = event_queue.get_next_event()
        logging.info(f"Processing event: {event.event_type} | Entity: {event.entity} | Arrival Time: {event.entity.arrival_time}")

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                rider_thickness_values.append(len(realization_graph.passive_riders))

                # Generate label and compatibility sets for the driver
                driver_label, label_to_set_map = generate_label(
                    False,
                    event.entity.location,
                    results,
                    lambda_i,
                    lambda_j,
                    mu_i
                )
                event.entity.label = driver_label
                event.entity.compatibility_set = label_to_set_map.get(driver_label, [])
                realization_graph.add_driver(event.entity)
                
                logging.info(f"Driver {event.entity.arrival_time} labeled {driver_label} | Location: {event.entity.location}")
                
                # Check for immediate matches with riders in the driver's compatibility set
                compatible_riders = [
                    rider for rider in realization_graph.passive_riders
                    if rider.location in event.entity.compatibility_set
                ]


                if compatible_riders:
                    best_rider = None
                    best_reward = -float('inf')
                    best_trip_distance = float('inf')

                    for rider in compatible_riders:
                        reward = rewards[event.entity.type][rider.type]
                        trip_distance = distance_matrix[event.entity.location][rider.location]

                        if reward > best_reward or (reward == best_reward and trip_distance < best_trip_distance):
                            best_rider = rider
                            best_reward = reward
                            best_trip_distance = trip_distance

                    if best_rider:
                        logging.info(f"Matched Driver {event.entity.arrival_time} | Location: {event.entity.location} with Rider {best_rider.arrival_time} | Location: {best_rider.location}")
                        perform_match(
                            realization_graph,
                            driver=event.entity,
                            rider=best_rider,
                            rewards=rewards,
                            distance_matrix=distance_matrix,
                            event_queue=event_queue,
                        )

            elif isinstance(event.entity, Rider):
                driver_thickness_values.append(len(realization_graph.active_drivers))

                event.entity.label = 0
                realization_graph.add_rider(event.entity)
                
                logging.info(f"Rider {event.entity.arrival_time} added | Location: {event.entity.location}")
                
                # Try to find a compatible driver
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards, distance_matrix)

                if matched_driver:
                    logging.info(f"Matched Rider {event.entity.arrival_time} | Location: {event.entity.location} with Driver {matched_driver.arrival_time} | Location: {matched_driver.location}")
                    perform_match(
                        realization_graph,
                        driver=matched_driver,
                        rider=event.entity,
                        rewards=rewards,
                        distance_matrix=distance_matrix,
                        event_queue=event_queue,
                    )

        elif event.event_type == 'abandonment':
            abandonments += 1
            if isinstance(event.entity, Driver):
                if event.entity in realization_graph.active_drivers:
                    logging.info(f"Driver {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_driver(event.entity)
                else:
                    raise RuntimeError(f"Driver abandonment inconsistency: {event.entity} not in active_drivers.")

            elif isinstance(event.entity, Rider):
                if event.entity in realization_graph.passive_riders:
                    logging.info(f"Rider {event.entity.arrival_time} abandoned | Location: {event.entity.location} | Abandonment Time: {event.entity.abandonment_time}")
                    realization_graph.remove_rider(event.entity)
                else:
                    raise RuntimeError(f"Rider abandonment inconsistency: {event.entity} not in passive_riders.")

    logging.info("Event processing completed.")
    print(abandonments, "abandonments")


    print(f"Average rider-side thickness (drivers available at rider arrival): {mean(driver_thickness_values):.3f}")
    print(f"Average driver-side thickness (riders available at driver arrival): {mean(rider_thickness_values):.3f}")

    realization_graph.print_summary()

