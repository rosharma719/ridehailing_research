from gurobipy import Model, GRB
import numpy as np

def solve_RB(rewards, lambda_i, lambda_j, mu_i):
    """
    Solve the LP relaxation (RB) using Gurobi based on the given rewards, arrival rates, and abandonment rates.
    
    :param rewards: A dictionary of rewards for each active-passive type pair.
    :param lambda_i: A dictionary of arrival rates for active types.
    :param lambda_j: A dictionary of arrival rates for passive types.
    :param mu_i: A dictionary of abandonment rates for active types.
    :return: A dictionary with the flow matrix, abandonment rates, and unmatched vertices.
    """
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types

    # Initialize Gurobi model for RB
    model = Model("RB_Model")

    # Decision variables for flow rates between active and passive types
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)
    x_a = model.addVars(I, name="x_a", lb=0, vtype=GRB.CONTINUOUS)  # Abandonment variables

    # Flow conservation constraints for active types
    for i in I:
        model.addConstr(
            sum(x[i, j] for j in J) + x_a[i] == lambda_i.get(i, 0),
            name=f"Flow_Constraint_{i}"
        )

    # Abandonment rate constraint for each active-passive pair
    for i in I:
        for j in J:
            model.addConstr(
                mu_i[i] * x[i, j] <= lambda_j[j] * x_a[i],
                name=f"Abandonment_Constraint_{i}_{j}"
            )

    # Objective function: Maximize total reward
    model.setObjective(
        sum(rewards[i][j] * x[i, j] for i in I for j in J),
        GRB.MAXIMIZE
    )

    # Optimize the model
    model.optimize()

    # Prepare the results
    flow_matrix = np.zeros((len(I), len(J)))
    abandonment = {}
    unmatched = {}

    if model.status == GRB.OPTIMAL:
        x_values = model.getAttr('x', x)
        x_a_values = model.getAttr('x', x_a)
        optimal_reward = model.ObjVal

        # Printing results with clear demarcation
        print("\n--- Optimal Flow Rates (RB Solution) ---")
        
        # Print flow rates x[i,j]
        print("\n--- Flow Rates x[i,j] ---")
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                flow_value = x_values[i, j]
                flow_matrix[i_idx, j_idx] = flow_value
                print(f"x[{i}, {j}] (Flow from Active {i} to Passive {j}): {flow_value:.4f}")
            print("-" * 40)

        # Print abandonment rates x_a[i]
        print("\n--- Abandonment Rates x_a[i] ---")
        for i in I:
            abandonment[i] = x_a_values[i]
            print(f"x_a[{i}] (Abandonment for Active {i}): {abandonment[i]:.4f}")
            print("-" * 40)

        # Compute unmatched vertices
        print("\n--- Unmatched Vertices ---")
        for i in I:
            unmatched[i] = lambda_i[i] - sum(flow_matrix[I.index(i), :]) - abandonment[i]
            print(f"Unmatched for {i}: {unmatched[i]:.4f}")
            print("-" * 40)

        print(f"\n--- Maximum Reward: {optimal_reward:.4f} ---\n")
    else:
        print("No optimal solution found for RB.")
    
    # Return the flow matrix, abandonment rates, and unmatched vertices in a dictionary
    return {
        'flow_matrix': flow_matrix,
        'abandonment': abandonment,
        'unmatched': unmatched
    }

def solve_QB(rewards, lambda_i, lambda_j, mu_i):
    """
    Solve the QCLP (QB) using Gurobi based on the given rewards, arrival rates, and abandonment rates.
    
    :param rewards: A dictionary of rewards for each active-passive type pair.
    :param lambda_i: A dictionary of arrival rates for active types.
    :param lambda_j: A dictionary of arrival rates for passive types.
    :param mu_i: A dictionary of abandonment rates for active types.
    :return: A dictionary with the flow matrix, abandonment rates, and unmatched vertices.
    """
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types

    # Initialize Gurobi model for QB
    model = Model("QB_Model")

    # Decision variables for flow rates between active and passive types
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)
    x_a = model.addVars(I, name="x_a", lb=0, vtype=GRB.CONTINUOUS)  # Abandonment variables

    # Flow conservation constraints for active types
    for i in I:
        model.addConstr(
            sum(x[i, j] for j in J) + x_a[i] == lambda_i.get(i, 0),
            name=f"Flow_Constraint_{i}"
        )

    # Abandonment rate constraints for each active type and passive type pair
    for i in I:
        for j in J:
            model.addConstr(
                mu_i[i] * x[i, j] <= lambda_j[j] * x_a[i],
                name=f"Abandonment_Constraint_{i}_{j}"
            )

    # Objective function: Maximize total reward
    model.setObjective(
        sum(rewards[i][j] * x[i, j] for i in I for j in J),
        GRB.MAXIMIZE
    )

    # Optimize the model
    model.optimize()

    # Prepare the results
    flow_matrix = np.zeros((len(I), len(J)))
    abandonment = {}
    unmatched = {}

    if model.status == GRB.OPTIMAL:
        x_values = model.getAttr('x', x)
        x_a_values = model.getAttr('x', x_a)
        optimal_reward = model.ObjVal

        # Printing results with clear demarcation
        print("\n--- Optimal Flow Rates (QB Solution) ---")

        # Print flow rates x[i,j]
        print("\n--- Flow Rates x[i,j] ---")
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                flow_value = x_values[i, j]
                flow_matrix[i_idx, j_idx] = flow_value
                print(f"x[{i}, {j}] (Flow from Active {i} to Passive {j}): {flow_value:.4f}")
            print("-" * 40)

        # Print abandonment rates x_a[i]
        print("\n--- Abandonment Rates x_a[i] ---")
        for i in I:
            abandonment[i] = x_a_values[i]
            print(f"x_a[{i}] (Abandonment for Active {i}): {abandonment[i]:.4f}")
            print("-" * 40)

        # Compute unmatched vertices
        print("\n--- Unmatched Vertices ---")
        for i in I:
            unmatched[i] = lambda_i[i] - sum(flow_matrix[I.index(i), :]) - abandonment[i]
            print(f"Unmatched for {i}: {unmatched[i]:.4f}")
            print("-" * 40)

        print(f"\n--- Maximum Reward: {optimal_reward:.4f} ---\n")
    else:
        print("No optimal solution found for QB.")
    
    # Return the flow matrix, abandonment rates, and unmatched vertices in a dictionary
    return {
        'flow_matrix': flow_matrix,
        'abandonment': abandonment,
        'unmatched': unmatched
    }
