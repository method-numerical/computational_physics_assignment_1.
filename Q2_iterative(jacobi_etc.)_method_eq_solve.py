#iterative method for solving equations (jacobi, gauss seidel, relaxation, conjugate gradient)
#I did not find direct library functions for these, so I had to implement direct algorithm.

import numpy as np
import scipy.sparse.linalg

#defining matrix A and b where Ax=b.
A=np.array([[0.2,0.1,1,1,0],
            [0.1,4,-1,1,-1],
            [1,-1,60,0,-2],
            [1,1,0,8,4],
            [0,-1,-2,4,700]])
b=np.array([[1,2,3,4,5]])

#correct solution
x_0=np.array([[7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163]])

#jacobi method
#taking initial solution as 0
x=np.zeros(5)
i=1
while i>=1:
    A1=np.diag(A)
    A2=A-np.diagflat(A1)
    x=(b-np.dot(x,A2))/A1
    if np.amax(abs(x_0-x))<0.01:
        break
    i=i+1
print("using jacobi method solution x=\n",x)
print("number of iteration used in jacobi method =",i,".\n\n")


#gauss siedel method
#taking initial solution as 0 vector
x=np.zeros(5)
i=1
while i>=1:
    A1=np.diag(A)
    A2=A-np.diagflat(A1)
    for j in range(5):
        x[j]=(b[0,j]-np.dot(x,A2[j]))/A1[j]
    if np.amax(abs(x-x_0))<0.01:
        break
    i=i+1
print("using gauss seidel method solution x=\n",x)
print("numbers of iteration used in gauss seidel method =",i,".\n\n")

#relaxation method
#taking initial solution as 0 vector
x=np.zeros(5)
w=1.25
i=1
while i>=1:
    A1=np.diag(A)
    A2=A-np.diagflat(A1)
    for j in range(5):
        r_j=b[0,j]-np.dot(x,A2[j])-A1[j]*x[j]
        x[j]=x[j]+w*(r_j/A1[j])
    if np.amax(abs(x-x_0))<0.01:
        break
    i=i+1
print("using relaxation method solution x=\n",x)
print("numbers of iteration used in relaxation method =",i,".\n\n")

#conjugate gradient method
#taking initial solution as 0 vector
x=np.zeros(5)
i=1
while i>=1:
    v=b-np.dot(x,A)
    z=np.dot(v,A)
    t=np.dot(v,np.transpose(v))/np.dot(v,np.transpose(z))
    x=x+t*v
    if np.amax(abs(x-x_0))<0.01:
        break
    i=i+1
print("using conjugate  gradient method solution x=\n",x)
print("numbers of iteration used in conjugate gradient method =",i,".\n\n")

print("----------------conclusion----------------\n\n")

print("relaxation is fastest for w>1\n")
print("gauss seidel is faster than jacobi\n")
print("conjugate gradient method is slowest for solving system of linear equations.")




#some scipy library function of slight other variance.
#relaxation method
#no control of w
#number of iterations can not be known easily,
import scipy.sparse.linalg as l
k=1
while k>=1:
    x_r,y=l.gmres(A,b[0],maxiter=k)
    if np.max(abs((x_r-x_0)/x_0))<0.0001:
        break
    else:
        k=k+1
print("\n\nsolution using scipy relaxation method:",x_r)
print("number of iterations:",k,"....")
