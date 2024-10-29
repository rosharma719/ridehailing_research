from eventgenerator import *
from utils import *

def greedy_auto_label(event_queue, rewards, flow_matrix):
    """
    Process events and generate labels for drivers and riders using the flow matrix \tilde{x}_{i,j}.
    
    :param event_queue: The queue of events (arrival, abandonment)
    :param rewards: The reward matrix for matches
    :param flow_matrix: The flow matrix \tilde{x}_{i,j} from the QB optimization
    """
    
    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, Driver):
                # Get the flow rates \tilde{x}_{i,j} for this driver node
                tildex_i_j = obtain_tildex(flow_matrix, event.entity.location)

                # Generate label for driver based on the flow matrix
                driver_label = generate_label(False, event.entity.location, flow_matrix, tildex_i_j, realization_graph.passive_riders, realization_graph.active_drivers)
                event.entity.label = driver_label  # Assign label to the driver
                realization_graph.add_driver(event.entity)

            # In greedy_auto_label function, add this in rider arrival
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
    
    :param event_queue: The event queue where events are stored
    :param entity: The entity (driver or rider) whose abandonment event needs to be removed
    """
    # Create a new queue to hold events that are not related to the matched entity's abandonment
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




def optimal_offline(event_queue): 
    print("hi")