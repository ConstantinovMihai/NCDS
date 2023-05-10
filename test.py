import cvxpy as cp
import numpy as np
import control

A = np.array([[-1,1], [-1,-1]])
Q = np.eye(2)

P1 = cp.Variable((2,2), PSD=True)
P2 = cp.Variable((2,2), PSD=True)

cons1 = [A.T @ P1 @ A - P1 + Q << 0]
cons2 = [A.T @ P2 @ A - P2     << 0]

prob1 = cp.Problem(cp.Minimize(0), cons1)
prob1.solve()

prob2 = cp.Problem(cp.Minimize(0), cons2)
prob2.solve()

print(P1.value)
print(P2.value)
