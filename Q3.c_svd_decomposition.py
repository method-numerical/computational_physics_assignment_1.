#A(m,n)=U(m,m)*S(m,n)*V_T(n,n) :parenthesis shows shape of matrix.
#code for singular value decomposition of a matrix

import time as tm
import numpy as np
import numpy.linalg as nl
np.set_printoptions(precision=4)

def sing_val_dec(A):
    m=len(A)                #nuumber of rows
    n=len(A[0])             #number of columns

    #constructing S matrix
    S=np.zeros((m,n))       #initializing S
    E,F=nl.eigh(np.dot(np.transpose(A),A))
    L=np.argsort(E)[::-1]   #to sort eigenvalues in descending order
    E=E[L]
    F=F[:,L]
    for i in range(min(m,n)):
        S[i,i]=np.sqrt(E[i])


    #constructing V matrix
    V=np.zeros((n,n))       #initializing V
    for i in range(n):
        V[:,i]=F[:,i]


    #constructing U matrix
    U=np.zeros((m,m))       #initializing U
    for i in range(min(m,n)):
        U[:,i]=np.dot(A,V[:,i])/S[i,i]
    if m-n<1:
        True
    else:
        for j in range(n,m):
            U[:,j]=np.zeros(m)
            U[:,j][0]=1
            p=np.zeros(m)
            for k in range(j):
                p=p+np.dot(np.transpose(U[:,k]),U[:,j])*U[:,k]
            U[:,j]=U[:,j]-p
            U[:,j]=U[:,j]/nl.norm(U[:,j])

    return U,S,np.transpose(V)

#using above code to calculate svd decomposition of given A.
A=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])

#monitoring the time in nanosecond precision
t1=tm.perf_counter_ns()
U_code,S_code,V_T_code=sing_val_dec(A)
t2=tm.perf_counter_ns()

print("U(defined code)=\n",U_code)
print("\nS(defined code)=\n",S_code)
print("\nV_T(defined code)=\n",V_T_code)
print("\ntime required using defined code=",t2-t1,"nano second.")

#comparison with numpy library function
t3=tm.perf_counter_ns()
U,S,V_T=nl.svd(A)
t4=tm.perf_counter_ns()

#U matrix differ from cases to cases, hence not expected to be same.
print("\nU(library function)=\n(due to high accuracy issue it may differ from above coded value)\n",U)
print("\nS(library function)=\n(numpy displayed only diagonal elements of S matrix)\n",S)
print("\nV_T(library function)=\n(due to high accuracy issue it may differ from above coded value)\n",V_T)
print("\ntime required using library function=",t4-t3,"nano second.")