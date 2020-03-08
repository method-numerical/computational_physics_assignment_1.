import numpy as np
import numpy.linalg as nl

#defining the matrix elements
A=np.array([[1,0.67,0.33],[0.45,1,0.55],[0.67,0.33,1]])
B=np.array([2,2,2])

X=nl.solve(A,B)

print('solution:',X)
