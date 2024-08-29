from gurobipy import Model, GRB
import numpy as np

# Set the Gurobi license path
import os
os.environ['GRB_LICENSE_FILE'] = r'C:\\Users\\kragg\\OneDrive\\Documents\\Code\\Licenses\\gurobi.lic'

def solve_RB(rewards, lambda_i, lambda_j):
    """
    Solve the LP relaxation (RB) using Gurobi based on the given rewards and arrival rates.
    
    :param rewards: A dictionary of rewards for each active-passive type pair.
    :param lambda_i: A dictionary of arrival rates for active types.
    :param lambda_j: A dictionary of arrival rates for passive types.
    :return: A NumPy array representing the flow matrix of optimal match rates.
    """
    # Extract active and passive types from the rewards dictionary
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types

    # Initialize Gurobi model for RB
    model = Model("RB_Model")

    # Decision variables for flow rates between active and passive types
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)

    # Flow conservation constraints for active types
    for i in I:
        model.addConstr(sum(x[i, j] for j in J) <= lambda_i[i],
                        name=f"Flow_Constraint_{i}")

    # Passive arrival constraints for passive types
    for j in J:
        model.addConstr(sum(x[i, j] for i in I) <= lambda_j[j],
                        name=f"Arrival_Constraint_{j}")

    # Objective function: Maximize total reward
    model.setObjective(sum(rewards[i][j] * x[i, j] for i in I for j in J),
                       GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Prepare the flow matrix
    flow_matrix = np.zeros((len(I), len(J)))

    # Retrieve and print results with clear labels
    if model.status == GRB.OPTIMAL:
        x_values = model.getAttr('x', x)
        optimal_reward = model.ObjVal
        print("Optimal Flow Rates (RB Solution):")
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                flow_value = x_values[i, j]
                flow_matrix[i_idx, j_idx] = flow_value
                print(f"Flow from {i} to {j}: {flow_value:.4f}")
        print(f"Maximum Reward: {optimal_reward:.4f}\n")
    else:
        print("No optimal solution found for RB.")
    
    # Return the flow matrix
    return flow_matrix