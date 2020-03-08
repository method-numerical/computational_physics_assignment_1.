import numpy as np
import numpy.linalg as nl

A=np.array([[5,-2],[-2,8]])

_A=A
i=0
while i>=0:
    Q,R=np.linalg.qr(_A)
    B=R@Q
    #iteration will stop when _A=B
    if (_A==B).all(): #this is "True" iff all elements of _A, B are equal
        break;
    else:
        _A=B;
    i+=1
e_code=np.diag(_A)
print("eigen-values using code:",e_code)
print("it is in descending order of eigen-values.")

#comparing with numpy library function
#it will sort eigen-vlaues in ascending order
e_lib_val,e_lib_vec=nl.eigh(A)
print("\neigen-values using numpy function:",e_lib_val)
print("numpy.linalg.eigh sorts the eigen-values in ascending order.")