from gurobipy import Model, GRB
import numpy as np

# Set the Gurobi license path
import os
os.environ['GRB_LICENSE_FILE'] = r'C:\\Users\\kragg\\OneDrive\\Documents\\Code\\Licenses\\gurobi.lic'

def solve_RB(rewards, lambda_i, lambda_j, mu_i):
    """
    Solve the LP relaxation (RB) using Gurobi based on the given rewards, arrival rates, and abandonment rates.
    
    :param rewards: A dictionary of rewards for each active-passive type pair.
    :param lambda_i: A dictionary of arrival rates for active types.
    :param lambda_j: A dictionary of arrival rates for passive types.
    :param mu_i: A dictionary of abandonment rates for active types.
    :return: A NumPy array representing the flow matrix of optimal match rates.
    """
    # Extract active and passive types from the rewards dictionary
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types

    # Initialize Gurobi model for RB
    model = Model("RB_Model")
    model.setParam('Presolve', 0) # Disable presolve 

    # Decision variables for flow rates between active and passive types
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)
    x_a = model.addVars(I, name="x_a", lb=0, vtype=GRB.CONTINUOUS)  # Abandonment variables

    # Flow conservation constraints for active types
    for i in I:
        model.addConstr(
            sum(x[i, j] for j in J) + x_a[i] == lambda_i.get(i, 0),
            name=f"Flow_Constraint_{i}"
        )

    # Abandonment rate constraint for each active type
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

    # Prepare the flow matrix
    flow_matrix = np.zeros((len(I), len(J)))

    print("Model variable:")
    print(model)
    print("Model constraints:")
    print(model.getConstrs())
    print("Model objective:")
    print(model.getObjective())
    model.write("model.lp")

    # Retrieve and print results with clear labels
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
            print(f"x_a[{i}] (Abandonment for Active {i}): {x_a_values[i]:.4f}")
            print("-" * 40)

        print(f"\n--- Maximum Reward: {optimal_reward:.4f} ---\n")
    else:
        print("No optimal solution found for RB.")
    
    # Return the flow matrix
    return flow_matrix

# Example usage
if __name__ == "__main__":
    # Test values (these should be provided or generated appropriately)
    rewards = {
        'Active Driver Node 0': {'Passive Rider Node 0': 2, 'Passive Rider Node 1': 1},
        'Active Driver Node 1': {'Passive Rider Node 0': 1, 'Passive Rider Node 1': 2}
    }
    lambda_i = {'Active Driver Node 0': 0.3, 'Active Driver Node 1': 0.3}
    lambda_j = {'Passive Rider Node 0': 0.4, 'Passive Rider Node 1': 0.4}
    mu_i = {'Active Driver Node 0': 0.2, 'Active Driver Node 1': 0.2}

    # Solve RB and get the flow matrix
    flow_matrix = solve_RB(rewards, lambda_i, lambda_j, mu_i)
