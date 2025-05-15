import json
import os
import logging
import numpy as np
import utils
import GurobiQB as qb
import eventgenerator as eg
import matchers
from offline_solver import solve_offline_optimal_with_markov
from logchecker import parse_log_file


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def configure_logging():
    logging.basicConfig(
        filename="logs.txt",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG
    )


def load_matrices(config, num_nodes, reward_value):
    if config.get("use_custom_matrices", False):
        adj_path = os.path.join(os.path.dirname(__file__), config["custom_adjacency_path"])
        reward_path = os.path.join(os.path.dirname(__file__), config["custom_reward_path"])

        print(f"\nUSING CUSTOM MATRICES:\n  Adjacency: {adj_path}\n  Reward: {reward_path}")
        adjacency = np.load(adj_path)
        with open(reward_path, "r") as f:
            rewards = json.load(f)
        return adjacency, rewards, adjacency, rewards

    grid_matrix = utils.generate_grid_adjacency_matrix(num_nodes)
    ring_matrix = utils.generate_ring_adjacency_matrix(num_nodes)
    reward_structure = config["reward_structure"]
    distance_penalty = config["distance_penalty"]

    if reward_structure == "divided":
        grid_rewards = utils.divided_adjacency_to_rewards(grid_matrix, reward_value, config["grid_denom"])
        ring_rewards = utils.divided_adjacency_to_rewards(ring_matrix, reward_value, config["ring_denom"])
    elif reward_structure == "standard":
        grid_rewards = utils.adjacency_to_rewards(grid_matrix, reward_value, distance_penalty)
        ring_rewards = utils.adjacency_to_rewards(ring_matrix, reward_value, distance_penalty)
    elif reward_structure == "increasing_distance_penalty":
        grid_rewards = utils.increasing_distance_penalty(grid_matrix, reward_value)
        ring_rewards = utils.increasing_distance_penalty(ring_matrix, reward_value)
    else:
        raise ValueError("Invalid reward structure. Choose from: 'divided', 'standard', 'increasing_distance_penalty'")

    return grid_matrix, grid_rewards, ring_matrix, ring_rewards


def load_rates(config, num_nodes, rate_riders, rate_drivers):
    driver_node_names = [f"Active Driver Node {i}" for i in range(num_nodes)]
    rider_node_names = [f"Passive Rider Node {i}" for i in range(num_nodes)]

    if config.get("use_custom_rates", False):
        with open(os.path.join(os.path.dirname(__file__), config["custom_rate_path"])) as f:
            custom_rates = json.load(f)
        lambda_i = {name: custom_rates[name]["arrival"] for name in driver_node_names}
        lambda_j = {name: custom_rates[name]["arrival"] for name in rider_node_names}
        mu_i = {name: custom_rates[name]["sojourn"] for name in driver_node_names}
    else:
        lambda_i = utils.generate_param_normal_distribution(driver_node_names, mean=rate_drivers, std_dev=0.2, min_val=0.05, max_val=2, seed=42)
        lambda_j = utils.generate_param_normal_distribution(rider_node_names, mean=rate_riders, std_dev=0.2, min_val=0.05, max_val=2, seed=101)
        mu_i = utils.generate_param_normal_distribution(driver_node_names, mean=1, std_dev=0.5, min_val=0.05, max_val=4, seed=7)

    return lambda_i, lambda_j, mu_i


def run_matchers(event_queue, rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix, topology_name, thickness_floor):
    print(f"\n\n{topology_name.upper()}\n")
    for matcher_name, matcher_fn in [
        ("GREEDY AUTO-LABEL", lambda q, *args: matchers.greedy_auto_label(q, *args, explicitly_remove_riders=True)),
        ("GREEDY", matchers.greedy_matcher),
        ("GREEDY AUTO-LABEL NONPERISHING", matchers.nonperish_bidirectional_GAL),
    ]:
        print(f"\n{matcher_name}\n")
        matcher_fn(event_queue.clone(), rewards, QB_results, lambda_i, lambda_j, mu_i, adjacency_matrix)

    print("\nOFFLINE OPTIMAL SOLUTION\n")
    solve_offline_optimal_with_markov(event_queue.clone(), rewards)


def main():
    config = load_config()
    configure_logging()

    num_nodes = config["num_nodes"]
    reward_value = num_nodes + 1
    simulation_time = config["simulation_time"]
    rate_riders = config["rate_riders"]
    rate_drivers = config["rate_drivers"]
    sojourn_rate_riders = config["sojourn_rate_riders"]
    sojourn_rate_drivers = config["sojourn_rate_drivers"]

    topology_mode = config["topology_mode"]
    thickness_floor = config["thickness_floor"]

    grid_matrix, grid_rewards, ring_matrix, ring_rewards = load_matrices(config, num_nodes, reward_value)
    lambda_i, lambda_j, mu_i = load_rates(config, num_nodes, rate_riders, rate_drivers)

    QB_results_grid = qb.solve_QB(grid_rewards, lambda_i, lambda_j, mu_i) if topology_mode in ["grid", "both"] or config.get("use_custom_matrices", False) else None
    QB_results_ring = qb.solve_QB(ring_rewards, lambda_i, lambda_j, mu_i) if topology_mode in ["ring", "both"] else None

    event_queue = eg.EventQueue()
    eg.generate_events(event_queue, rate_riders, rate_drivers, sojourn_rate_riders, sojourn_rate_drivers, num_nodes, simulation_time)
    num_arrivals, num_abandonments = utils.count_events(event_queue)
    print(f"Total Arrivals: {num_arrivals}, Total Abandonments: {num_abandonments}")

    if config.get("use_custom_matrices", False):
        run_matchers(event_queue, grid_rewards, QB_results_grid, lambda_i, lambda_j, mu_i, grid_matrix, "custom", thickness_floor)
    else:
        if topology_mode in ["grid", "both"]:
            run_matchers(event_queue, grid_rewards, QB_results_grid, lambda_i, lambda_j, mu_i, grid_matrix, "grid", thickness_floor)
        if topology_mode in ["ring", "both"]:
            run_matchers(event_queue, ring_rewards, QB_results_ring, lambda_i, lambda_j, mu_i, ring_matrix, "ring", thickness_floor)

    print(parse_log_file("logs.txt"))


if __name__ == "__main__":
    main()
