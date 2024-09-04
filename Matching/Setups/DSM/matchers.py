import eventgenerator as eg
from utils import *

def greedy_auto_label(event_queue, rewards):
    realization_graph = RealizationGraph()
    printStuff = False

    while not event_queue.is_empty():
        event = event_queue.get_next_event()

        if event.event_type == 'arrival':
            if isinstance(event.entity, eg.Driver):
                realization_graph.add_driver(event.entity)
            elif isinstance(event.entity, eg.Rider):
                realization_graph.add_rider(event.entity)
                matched_driver = realization_graph.find_driver_for_rider(event.entity, rewards)
                if matched_driver and printStuff:
                    print(f"Matched {event.entity.type} at {event.entity.location} with {matched_driver.type}")
                elif printStuff:
                    print(f"No available drivers for {event.entity.type} at {event.entity.location}. Rider abandoned.")
        
        elif event.event_type == 'abandonment':
            if isinstance(event.entity, eg.Driver):
                if event.entity in realization_graph.active_drivers:
                    realization_graph.remove_driver(event.entity)
                    if printStuff:
                        print(f"{event.entity.type} at {event.entity.location} left the system")
            elif isinstance(event.entity, eg.Rider):
                if event.entity in realization_graph.passive_riders:
                    realization_graph.remove_rider(event.entity)
                    if printStuff:
                        print(f"{event.entity.type} at {event.entity.location} left the system")

    print("Event processing completed.")
    realization_graph.print_summary()


def optimal_offline(event_queue): 
    print("hi")

