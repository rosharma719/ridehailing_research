import random
import numpy as np
from scipy.optimize import linear_sum_assignment


def generate_random_pickup_dropoff(n, num_pickups):
    
    results = []
    for _ in range(num_pickups):
        pickup_x = random.uniform(0, n)
        pickup_y = random.uniform(0, n)
        dropoff_x = random.uniform(0, n)
        dropoff_y = random.uniform(0, n)
        
        horizontal_distance = abs(dropoff_x - pickup_x)
        vertical_distance = abs(dropoff_y - pickup_y)
        total_distance = horizontal_distance + vertical_distance
        
        results.append((pickup_x, pickup_y, dropoff_x, dropoff_y, total_distance))
    
    return results

def generate_random_drivers(n, num_drivers):
    results = []
    for _ in range(num_drivers):
        drivers_x = random.uniform(0, n)
        drivers_y = random.uniform(0, n)
        results.append((drivers_x, drivers_y))

    return results

def calculate_total_distance(driver, pickup, dropoff):
    driver_to_pickup = np.sqrt((driver[0] - pickup[0])**2 + (driver[1] - pickup[1])**2)
    pickup_to_dropoff = np.sqrt((pickup[0] - dropoff[0])**2 + (pickup[1] - dropoff[1])**2)
    return driver_to_pickup + pickup_to_dropoff

def assign_drivers_to_pickups(drivers, pickups):
    
    num_drivers = len(drivers)
    num_pickups = len(pickups)

    cost_matrix = np.zeros((num_drivers, num_pickups))
    for i in range(num_drivers):
        for j in range(num_pickups):
            pickup_x, pickup_y, dropoff_x, dropoff_y, _ = pickups[j]
            cost_matrix[i][j] = calculate_total_distance(drivers[i], (pickup_x, pickup_y), (dropoff_x, dropoff_y))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = {row: col for row, col in zip(row_ind, col_ind)}

    return assignment

def print_ride_times(assignments, drivers, pickups):
    for driver_index, pickup_index in assignments.items():
        driver = drivers[driver_index]
        pickup = pickups[pickup_index]
        total_distance = calculate_total_distance(driver, (pickup[0], pickup[1]), (pickup[2], pickup[3]))
        print(f"Driver {driver_index} to Pickup {pickup_index}: Total travel time = {total_distance:.2f} minutes")


region_size = 10  # Size of the region (10x10)
num_pickups = 5     # Generate 5 pickups
num_drivers = 5
generated_pickups = generate_random_pickup_dropoff(region_size, num_pickups)
generated_drivers = generate_random_drivers(region_size, num_drivers)

assignments = assign_drivers_to_pickups(generated_drivers, generated_pickups)
print(assignments)


# Print ride times using the previously generated data
print_ride_times(assignments, generated_drivers, generated_pickups)