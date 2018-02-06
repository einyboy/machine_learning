import numpy as np
import matplotlib.pyplot as plt

def find(x):
    return np.nonzero(x)[0]
    
n = 1000
p = 2
omega = np.array([1.,.5])*5

n1 = int(n/2)
X = np.vstack(( np.random.randn(n1,2), np.random.randn(n1,2)+np.ones([n1,1])*omega ))
y = np.vstack(( np.ones([n1,1]), -np.ones([n1,1])))
I = find(y==-1)
J = find(y==1)
plt.clf
plt.plot(X[I,0], X[I,1], '.')
plt.plot(X[J,0], X[J,1], '.')
plt.axis('equal');
plt.show()
 
t = np.linspace(-3,3,255).transpose()
plt.clf
plt.plot(t, t>0)
plt.plot(t, np.log(1+np.exp(t)))
plt.plot(t, np.maximum(t,0) )
plt.axis('tight');
plt.legend(['Binary', 'Logistic', 'Hinge']);
plt.show()

def L(s,y):  return 1/n * sum( np.log( 1 + np.exp(-s*y) ) )
def E(w,X,y): return L(X.dot(w),y);

def theta(v): return 1 / (1+np.exp(-v))
def nablaL(s,r): return - 1/n * y * theta(-s * y)
def nablaE(w,X,y): return X.transpose().dot( nablaL(X.dot(w),y) )

def AddBias(X): return np.hstack(( X, np.ones((np.size(X,0),1)) ))
w = np.zeros((p+1,1))

tau = .8; # here we are using a fixed tau
w = w - tau * nablaE(w,AddBias(X),y)
np.linalg.norm(X)
tau_max = 2/(1/4 * np.linalg.norm(AddBias(X), 2)**2 )
print(tau_max)

q = 201
tx = np.linspace( X[:,0].min(), X[:,0].max(),num=q) 
ty = np.linspace( X[:,1].min(), X[:,1].max(),num=q) 
[B,A] = np.meshgrid( ty,tx )
G = np.vstack([A.flatten(), B.flatten()]).transpose()

Theta = theta(AddBias(G).dot(w))
Theta = Theta.reshape((q,q))

plt.clf
plt.imshow(Theta.transpose(), origin="lower",  extent=[tx.min(),tx.max(),ty.min(),ty.max()])
plt.axis('equal')
plt.plot(X[I,0], X[I,1], '.')
plt.plot(X[J,0], X[J,1], '.')
plt.axis('off');
plt.show()