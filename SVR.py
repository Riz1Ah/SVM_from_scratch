

def visualize(x,y,w,b,e1,e2):
    
    for i in range(len(x)):
        
        if(abs(y[i]-w.T*np.transpose([x[i,:]])-1-b-e1[i])<10e-9):
            xp=x[i][0]
            yp=y[i]
        elif(abs(-y[i]+w.T*np.transpose([x[i,:]])-1+b-e2[i])<10e-9):
            xn=x[i][0]
            yn=y[i]
    
    lx=np.arange(-4,4,0.1)
    lyp=w*(lx-xp)+yp
    lyn=w*(lx-xn)+yn
    lyp=(np.array(lyp)).flatten()
    lyn=(np.array(lyn)).flatten()
    plt.figure()
    plt.plot(lx,(lyp+lyn)/2,'r-')
    plt.scatter(x[:,0],y,marker='o')
# =============================================================================
#     plt.xlim([-5,10])
#     plt.ylim([-15,0])
# =============================================================================
    
    



from sklearn.datasets import make_regression
dim=1
n_samp=100
(X_p,y_p)=make_regression(n_samples=n_samp,n_features=1,noise=10,random_state=10)
#(X_n,y_n)=make_blobs(n_samples=50,n_features=2,centers=1,cluster_std=1.05,random_state=1)
#y_n=2*(y_n-0.5)
import numpy as np

X_p=np.array(X_p)
#y_p=2*(y_p-0.5)

from matplotlib import pyplot as plt
#plt.scatter(X_n[:,0],X_n[:,1],marker='o',c=y_n)
#plt.scatter(X_p,y_p,marker='o')
#plt.plot(X_n[:,0],X_n[:,1],'rs',X_p[:,0],X_p[:,1],'b*')
import cvxpy as cvx
w=cvx.Variable(dim)
b=cvx.Variable()
e1=cvx.Variable(n_samp)
e2=cvx.Variable(n_samp)
C=1

constraint=[y_p[i]-((w.T*np.transpose([X_p[i,:]]))+b)-1-e1[i]<=0 for i in range(n_samp)]
constraint=constraint+[e1>=0 for i in range(n_samp)]
constraint=constraint+[-y_p[i]+((w.T*np.transpose([X_p[i,:]]))+b)-1-e2[i]<=0 for i in range(n_samp)]
constraint=constraint+[e2>=0 for i in range(n_samp)]
obj=cvx.Minimize(cvx.norm(w)+C*(cvx.sum_entries(e1)+cvx.sum_entries(e2)))
prob=cvx.Problem(obj,constraint)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", w.value, b.value,e1.value)
if type(w) == cvx.expressions.variables.variable.Variable: # These haven't yet been typecast
    w = w.value
    b = b.value
    e1 = e1.value
    e2 = e2.value
    

visualize(X_p,y_p,w,b,e1,e2)
#np.dot(w.T,[4.14,4.84])+b