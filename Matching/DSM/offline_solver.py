from scipy.optimize import linear_sum_assignment
import numpy as np
import eventgenerator as eg

def solve_offline_optimal_with_markov(event_queue, rewards):
    """
    Solve the offline optimal matching problem using a global optimization approach.

    Parameters:
    - event_queue: The queue of events (arrival, abandonment).
    - rewards: The reward matrix for matches.

    Returns:
    A dictionary containing the total reward, matchings, and statistics.
    """
    # Extract entities from the event queue
    events = list(event_queue.queue)
    agents = [event.entity for event in events if event.event_type == 'arrival']

    # Separate riders and drivers
    riders = [agent for agent in agents if isinstance(agent, eg.Rider)]
    drivers = [agent for agent in agents if isinstance(agent, eg.Driver)]

    # Debug: Log total riders and drivers
    total_riders_processed = len(riders)
    total_drivers_processed = len(drivers)

    # Create reward matrix for global matching
    reward_matrix = np.zeros((len(drivers), len(riders)))
    for i, driver in enumerate(drivers):
        for j, rider in enumerate(riders):
            # Check if their times overlap
            if driver.arrival_time < rider.abandonment_time and rider.arrival_time < driver.abandonment_time:
                reward_matrix[i, j] = rewards[driver.type][rider.type]
            else:
                reward_matrix[i, j] = 0  # No reward if times don't overlap

    # Solve the maximum-weight matching problem
    row_ind, col_ind = linear_sum_assignment(-reward_matrix)  # Negative for maximization

    # Calculate total reward and construct matchings
    total_reward = 0
    total_trip_distance = 0
    total_wait_time = 0
    matchings = []

    for i, j in zip(row_ind, col_ind):
        if reward_matrix[i, j] > 0:  # Only consider valid matches
            driver = drivers[i]
            rider = riders[j]
            reward = reward_matrix[i, j]
            wait_time = abs(driver.arrival_time - rider.arrival_time)
            trip_distance = abs(driver.location - rider.location)  # Assuming Manhattan distance

            total_reward += reward
            total_trip_distance += trip_distance
            total_wait_time += wait_time
            matchings.append((driver, rider, reward))

    # Generate summary statistics
    num_matches = len(matchings)
    avg_reward = total_reward / num_matches if num_matches > 0 else 0
    avg_trip_distance = total_trip_distance / num_matches if num_matches > 0 else 0
    avg_wait_time = total_wait_time / num_matches if num_matches > 0 else 0

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of Drivers Matched: {num_matches}")
    print(f"Number of Riders Matched: {num_matches}")
    print(f"Total Riders Processed: {total_riders_processed}")
    print(f"Total Drivers Processed: {total_drivers_processed}")
    print(f"Total Rewards: {total_reward:.2f}")
    print(f"Average Reward per Transaction: {avg_reward:.2f}")
    print(f"Average Trip Distance: {avg_trip_distance:.2f} units")
    print(f"Average Wait Time: {avg_wait_time:.2f} units\n\n")

