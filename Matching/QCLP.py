import mosek.fusion as mf
import numpy as np

def solve_qclp(T, r, lambd, mu):
    M = mf.Model("QCLP-QB")
    
    # Variables
    x = M.variable("x", [len(T), len(T)], mf.Domain.greaterThan(0.0))
    x_a = M.variable("x_a", len(T), mf.Domain.greaterThan(0.0))
    x_buf = M.variable("x_buf", len(T), mf.Domain.greaterThan(0.0))
    z = M.variable("z", [len(T), len(T)], mf.Domain.greaterThan(0.0))  # Auxiliary variable for product

    # Objective: Maximize total reward
    M.objective("total_reward", mf.ObjectiveSense.Maximize, 
                mf.Expr.sum(mf.Expr.mulElm(r, x)))
    
    # Flow balance constraints
    for i in range(len(T)):
        M.constraint(f"flow_balance_{i}", 
                     mf.Expr.add([
                         mf.Expr.sum(x.slice([i, 0], [i+1, len(T)])), 
                         mf.Expr.sum(x.slice([0, i], [len(T), i+1])), 
                         x_a.index(i).asExpr(), 
                         x_buf.index(i).asExpr()]), 
                     mf.Domain.equalsTo(lambd[i]))
    
    # Quadratic constraints
    for i in range(len(T)):
        for j in range(len(T)):
            # Introduce the auxiliary variable `z[i,j]` representing x_a[i] * (x_buf[j] + sum(x[k,j] for k in T))
            lhs = mf.Expr.mul(mu[i], x.index(i, j))
            rhs = mf.Expr.add(z.index(i, j), x_buf.index(j).asExpr())
            
            # Conic constraint representing the quadratic relationship
            M.constraint(mf.Expr.vstack(x_a.index(i).asExpr(), z.index(i, j), rhs), mf.Domain.inRotatedQCone())

    # Solve the model
    M.solve()
    
    # Extract the solution
    x_sol = x.level().reshape(len(T), len(T))
    x_a_sol = x_a.level()
    x_buf_sol = x_buf.level()
    
    return x_sol, x_a_sol, x_buf_sol

# Example input
T = [0, 1]
r = np.array([[0.0, 2.0], [2.0, 0.0]])
lambd = np.array([3.0, 2.0])
mu = np.array([1.0, 1.5])

# Solve
x_sol, x_a_sol, x_buf_sol = solve_qclp(T, r, lambd, mu)
print("x:", x_sol)
print("x_a:", x_a_sol)
print("x_buf:", x_buf_sol)
