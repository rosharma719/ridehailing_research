import utils
import GurobiRBandQB as grb
import numpy as np

# Static parameters
num_nodes = 2
skip_prob = 0
extra_edges = 0
rate_riders = 0.4
rate_drivers = 0.3
sojourn_rate_riders = 0.5
sojourn_rate_drivers = 0.2


# Parameter ranges for iteration
reward_value_range = [num_nodes + 5, num_nodes + 7, num_nodes]
distance_penalty_range = [-2, -1, 0, 1, 2, 3]


# Script to find heterogeneous and nonzero flow rates
def find_heterogeneous_flows():
    # Generate a fixed adjacency matrix
    adj_matrix = utils.generate_imperfect_grid_adjacency_matrix(num_nodes, skip_prob, extra_edges)

    for reward_value in reward_value_range:
        for distance_penalty in distance_penalty_range:
            # Generate reward matrix
            rewards = utils.adjacency_to_rewards(adj_matrix, reward_value, distance_penalty)

            # Define arrival and abandonment rates
            lambda_i = {f"Active Driver Node {i}": rate_drivers for i in range(num_nodes)}
            lambda_j = {f"Passive Rider Node {i}": rate_riders for i in range(num_nodes)}
            mu_i = {f"Active Driver Node {i}": sojourn_rate_drivers for i in range(num_nodes)}

            # Solve QB optimization to get flow matrix
            result = grb.solve_QB(rewards, lambda_i, lambda_j, mu_i)
            QB_flow_matrix = result['flow_matrix']

            # Check for heterogeneity and nonzero flows
            tildex_i_j = [np.sum(QB_flow_matrix[i, :]) for i in range(num_nodes)]
            if len(set(tildex_i_j)) > 1 and np.sum(tildex_i_j) > 0:
                print(f"\nHeterogeneous and Nonzero Flows Found:")
                print(f"reward_value: {reward_value}, distance_penalty: {distance_penalty}")
                print(f"Flow rates: {tildex_i_j}")

if __name__ == "__main__":
    find_heterogeneous_flows()
