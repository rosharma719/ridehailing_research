
import gurobipy as gp
from gurobipy import Model, GRB
import numpy as np

def solve_RB(rewards, lambda_i, lambda_j, mu_i, save_lp=True):
    """
    Solve the RB optimization problem using Gurobi.
    
    Parameters:
    - rewards: Dictionary of rewards for each active-passive type pair
    - lambda_i: Dictionary of arrival rates for active types
    - lambda_j: Dictionary of arrival rates for passive types
    - mu_i: Dictionary of abandonment rates for active types
    - save_lp: Boolean flag to save the model in LP format
    
    Returns:
    Dictionary containing flow matrix, abandonment rates, and unmatched vertices
    """
    I = list(rewards.keys())  # Active types
    J = list(rewards[I[0]].keys())  # Passive types

    # Initialize Gurobi model
    model = Model("RB_Model")

    # Decision variables
    x = model.addVars(I, J, name="x", lb=0, vtype=GRB.CONTINUOUS)  # Flow variables x_{i,j}
    x_a = model.addVars(I, name="x_a", lb=0, vtype=GRB.CONTINUOUS)  # Abandonment variables x_{i,a}

    # Objective: Maximize total reward
    model.setObjective(
        sum(rewards[i][j] * x[i, j] for i in I for j in J),
        GRB.MAXIMIZE
    )

    # Flow conservation constraints for active nodes
    for i in I:
        model.addConstr(
            sum(x[i, j] for j in J) + x_a[i] == lambda_i[i],
            name=f"Flow_Constraint_Active_{i}"
        )

    # Flow conservation constraints for passive nodes
    for j in J:
        model.addConstr(
            sum(x[i, j] for i in I) <= lambda_j[j],
            name=f"Flow_Constraint_Passive_{j}"
        )

    # Abandonment rate constraints for each active-passive pair
    for i in I:
        for j in J:
            model.addConstr(
                mu_i[i] * x[i, j] <= lambda_j[j] * x_a[i],
                name=f"Abandonment_Constraint_{i}_{j}"
            )

    # Non-negativity constraints
    for i in I:
        for j in J:
            model.addConstr(x[i, j] >= 0, name=f"NonNegativity_{i}_{j}")
        model.addConstr(x_a[i] >= 0, name=f"NonNegativity_{i}_abandonment")

    # Save LP file if requested
    if save_lp:
        model.write("rb_model.lp")

    # Optimize the model
    model.optimize()

    # Process results
    results = {
        'flow_matrix': np.zeros((len(I), len(J))),
        'abandonment': {},
        'unmatched': {}
    }

    if model.status == GRB.OPTIMAL:
        # Extract solution values
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                results['flow_matrix'][i_idx, j_idx] = x[i, j].X
            results['abandonment'][i] = x_a[i].X

        # Calculate unmatched vertices
        for i in I:
            results['unmatched'][i] = lambda_i[i] - sum(results['flow_matrix'][I.index(i), :]) - results['abandonment'][i]

        # Print summary
        print("\n=== RB Optimization Results ===")
        print(f"Objective Value: {model.objVal:.4f}")
        print("\nFlow Matrix:")
        for i_idx, i in enumerate(I):
            for j_idx, j in enumerate(J):
                print(f"{i} -> {j}: {results['flow_matrix'][i_idx, j_idx]:.4f}")
        print("\nAbandonment Rates:")
        for i in I:
            print(f"{i}: {results['abandonment'][i]:.4f}")

    else:
        print("No optimal solution found")

    return results
