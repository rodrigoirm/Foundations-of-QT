import numpy as np
import cvxpy as cp
import mosek

n=4

A = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
B = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

X = cp.Variable((n,n))
constraints = [cp.trace(B@X)==1,cp.trace(X)==1,X >> 0]

prob = cp.Problem(cp.Maximize(cp.trace(A @ X)),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(np.round(X.value,decimals=2))
