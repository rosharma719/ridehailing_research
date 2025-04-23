import gurobipy as gp
from gurobipy import Model, GRB
import numpy as np

def solve_QB(rewards, lambda_i, lambda_j, mu_i, save_lp=True):
    """
    Solve the QB optimization problem using Gurobi.

    Parameters:
    - rewards: Dictionary of rewards for each active-passive type pair
    - lambda_i: Dictionary of arrival rates for active types
    - lambda_j: Dictionary of arrival rates for passive types
    - mu_i: Dictionary of abandonment rates for active types
    - save_lp: Boolean flag to save the model in LP format

    Returns:
    Dictionary containing flow matrix, abandonment rates, unmatched vertices, and other results.
    """
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types


    # Initialize Gurobi model for QB
    model = Model("QB_Model")

    # Decision variables
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)  # Flow variables x_{i,j}
    x_i = model.addVars(I, name="x_i", lb=0, vtype=GRB.CONTINUOUS)  # Buffer for active unmatched vertices x_i
    x_j = model.addVars(J, name="x_j", lb=0, vtype=GRB.CONTINUOUS)  # Buffer for passive unmatched vertices x_j
    x_a = model.addVars(I, name="x_a", lb=0, vtype=GRB.CONTINUOUS)  # Abandonment variables x_{i,a}

    # Objective: Maximize total reward
    model.setObjective(
        sum(rewards[i][j] * x[i, j] for i in I for j in J),
        GRB.MAXIMIZE
    )

    # Flow conservation constraints for active types
    for i in I:
        model.addConstr(
            sum(x[i, j] for j in J) + x_a[i] + x_i[i] == lambda_i[i],
            name=f"Flow_Conservation_{i}"
        )

    # Flow conservation constraints for passive types
    for j in J:
        model.addConstr(
            sum(x[i, j] for i in I) + x_j[j] == lambda_j[j],
            name=f"Passive_Flow_Conservation_{j}"
        )

    # Quadratic abandonment constraints for active-passive pairs
    for i in I:
        for j in J:
            model.addQConstr(
                mu_i[i] * x[i, j] <= (x_j[j] + sum(x[k, j] for k in I)) * x_a[i],
                name=f"Quadratic_Abandonment_{i}_{j}"
            )

    # Non-negativity constraints
    for i in I:
        for j in J:
            model.addConstr(x[i, j] >= 0, name=f"NonNegativity_{i}_{j}")
        model.addConstr(x_a[i] >= 0, name=f"NonNegativity_{i}_abandonment")
        model.addConstr(x_i[i] >= 0, name=f"NonNegativity_{i}_buffer")
    for j in J:
        model.addConstr(x_j[j] >= 0, name=f"NonNegativity_{j}_buffer")

    # Save LP file if requested
    if save_lp:
        model.write("qb_model.lp")

    # Optimize
    model.optimize()
    
    # Process results
    results = {
        'flow_matrix': np.zeros((len(I), len(J))),
        'abandonment': {},
        'passive_unmatched': {},
        'active_unmatched': {},
        'unmatched': {}
    }

    if model.status == GRB.OPTIMAL:
        # Extract solution values
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                results['flow_matrix'][i_idx, j_idx] = round(x[i, j].X, 4)
            results['abandonment'][i] = round(x_a[i].X, 4)
            results['active_unmatched'][i] = round(x_i[i].X, 4)

        for j in J:
            results['passive_unmatched'][j] = round(x_j[j].X, 4)

        # Calculate unmatched vertices
        for i in I:
            results['unmatched'][i] = round(
                lambda_i[i] - sum(results['flow_matrix'][I.index(i), :]) 
                - results['abandonment'][i] - results['active_unmatched'][i],
                4
            )

        # Print summary
        print("\n=== QB Optimization Results ===")
        print(f"Objective Value: {model.objVal:.4f}")
        print("\nFlow Matrix:")
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                print(f"{i} -> {j}: {results['flow_matrix'][i_idx, j_idx]:.4f}")
        print("\nAbandonment Rates:")
        for i in I:
            print(f"{i}: {results['abandonment'][i]:.4f}")
        print("\nPassive Unmatched (Buffer x_j):")
        for j in J:
            print(f"{j}: {results['passive_unmatched'][j]:.4f}")
        print("\nActive Unmatched (Buffer x_i):")
        for i in I:
            print(f"{i}: {results['active_unmatched'][i]:.4f}")
    else:
        print("No optimal solution found")

    return results
