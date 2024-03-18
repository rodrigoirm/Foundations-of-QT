import numpy as np
import cvxpy as cp
import mosek
n = 2
m = 4

A = np.array([[-2,-1],[-1,-3],[-1,0],[0,-1]])
b = np.array([-1,-1,0,0])
c_a = np.array([1,1])
c_b = np.array([-1,-1])
c_final = np.array([1,0])

x = cp.Variable(n)
prob_a = cp.Problem(cp.Minimize(c_a.T@x),[A@x<=b])
prob_a.solve(solver=cp.MOSEK)
print("\nThe optimal value is", prob_a.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob_a.constraints[0].dual_value)

prob_b = cp.Problem(cp.Minimize(c_b.T@x),[A@x<=b])
prob_b.solve(solver=cp.MOSEK)
print("\nThe optimal value is", prob_b.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob_b.constraints[0].dual_value)

prob_c = cp.Problem(cp.Minimize(c_final.T@x),[A@x<=b])
prob_c.solve(solver=cp.MOSEK)
print("\nThe optimal value is", prob_c.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob_c.constraints[0].dual_value)

