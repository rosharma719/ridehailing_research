import gurobipy as gp
from gurobipy import GRB

# Set the Gurobi license path
import os
os.environ['GRB_LICENSE_FILE'] = r'C:\\Users\\kragg\\OneDrive\\Documents\\Code\\Licenses\\gurobi.lic'

# Linear Programming Problem
def solve_lp():
    model = gp.Model("Linear_Programming")

    # Variables
    x1 = model.addVar(name="x1", lb=0)
    x2 = model.addVar(name="x2", lb=0)

    # Objective
    model.setObjective(x1 + 0.5 * x2, GRB.MAXIMIZE)

    # Constraint
    model.addConstr(x1 + x2 == 1, "c1")

    # Solve
    model.optimize()

    print("\nLinear Programming Solution:")
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')
    print(f'Objective Value: {model.objVal}')

# Quadratic Programming Problem 1
def solve_qp1():
    model = gp.Model("Quadratic_Programming_1")

    # Variables
    x1 = model.addVar(name="x1", lb=0)
    x2 = model.addVar(name="x2", lb=0)

    # Objective
    model.setObjective(x1 + 0.5 * x2, GRB.MAXIMIZE)

    # Constraints
    model.addConstr(x1 * x1 + x2 * x2 <= 1, "c1")

    # Solve
    model.optimize()

    print("\nQuadratic Programming 1 Solution:")
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')
    print(f'Objective Value: {model.objVal}')

# Quadratic Programming Problem 2
def solve_qp2():
    model = gp.Model("Quadratic_Programming_2")

    # Variables
    x1 = model.addVar(name="x1", lb=0)
    x2 = model.addVar(name="x2", lb=0)
    x3 = model.addVar(name="x3", lb=0)

    # Objective
    model.setObjective(x1 + x2 + x3, GRB.MAXIMIZE)

    # Constraints
    model.addConstr(x1 <= x2 * x3, "c1")

    # Solve
    model.optimize()

    print("\nQuadratic Programming 2 Solution:")
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')
    print(f'Objective Value: {model.objVal}')

if __name__ == "__main__":
    solve_lp()
    solve_qp1()
    solve_qp2()
