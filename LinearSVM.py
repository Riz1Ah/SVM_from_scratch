

def visualize(x,y,w,b):
    
    for i in range(len(x)):
        
        if(abs(w.T*np.transpose([x[i,:]])-1+b)<10e-9):
            xp=x[i][0]
            yp=x[i][1]
        elif(abs(w.T*np.transpose([x[i,:]])+1+b)<10e-9):
            xn=x[i][0]
            yn=x[i][1]
    
    lx=np.arange(-10,10,0.1)
    lyp=-w[0]/w[1]*(lx-xp)+yp
    lyn=-w[0]/w[1]*(lx-xn)+yn
    lyp=(np.array(lyp)).flatten()
    lyn=(np.array(lyn)).flatten()
    plt.figure()
    plt.plot(lx,lyn,'r-',lx,lyp,'r-')
    plt.scatter(x[:,0],x[:,1],marker='o',c=y)
# =============================================================================
#     plt.xlim([-5,10])
#     plt.ylim([-15,0])
# =============================================================================
    
    



from sklearn.datasets.samples_generator import make_blobs
dim=2
n_samp=100
(X_p,y_p)=make_blobs(n_samples=n_samp,n_features=2,centers=2,cluster_std=2,random_state=22)
#(X_n,y_n)=make_blobs(n_samples=50,n_features=2,centers=1,cluster_std=1.05,random_state=1)
#y_n=2*(y_n-0.5)
import numpy as np
X_p=np.array(X_p)
y_p=2*(y_p-0.5)

from matplotlib import pyplot as plt
# =============================================================================
# plt.scatter(X_n[:,0],X_n[:,1],marker='o',c=y_n)
#plt.scatter(X_p[:,0],X_p[:,1],marker='o',c=y_p)
#plt.plot(X_n[:,0],X_n[:,1],'rs',X_p[:,0],X_p[:,1],'b*')
# =============================================================================
import cvxpy as cvx
w=cvx.Variable(dim)
b=cvx.Variable()

constraint=[y_p[i]*((w.T*np.transpose([X_p[i,:]]))+b)>=1 for i in range(n_samp)]
obj=cvx.Minimize(cvx.norm(w))   
prob=cvx.Problem(obj,constraint)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", w.value, b.value)
if type(w) == cvx.expressions.variables.variable.Variable: # These haven't yet been typecast
    w = w.value
    b = b.value
    

visualize(X_p,y_p,w,b)
#np.dot(w.T,[4.14,4.84])+b