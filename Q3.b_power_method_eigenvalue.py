import numpy as np
import numpy.linalg as nl

def ratio_iter(A,x_0,y_0,i):
    p=y_0@nl.matrix_power(A,i+1)@x_0
    q=y_0@nl.matrix_power(A,i)@x_0
    return p/q

A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
y_0=np.array([1,0,0])
x_0=np.array([0,1,0])

i=1
while i>=1:
    x=ratio_iter(A,x_0,y_0,i+1)/ratio_iter(A,x_0,y_0,i)-1
    if abs(x)<0.01:
        break
    else:
        i=i+1

#dominant eigevalue and the corresponding eigenvector
e_val=ratio_iter(A,x_0,y_0,i+1)
e_vec=nl.matrix_power(A,i+1)@x_0
e_vec_mod=e_vec/e_vec[0]

print("dominant eigenvalue is ",e_val,".\n")
print("corresponding eigenvector (transposed) is ",np.transpose(e_vec_mod),".\n")