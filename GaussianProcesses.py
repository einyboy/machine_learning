
import numpy as np
import matplotlib.pyplot as plt

coefs = [6, -2.5, -2.4, -0.1, 0.2, 0.03]

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef*(x**exp)
    return total

def k(xs, ys, sigma = 1, l = 1):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return sigma**2 * np.exp(-((dx/l)**2)/2)
    
def m(x):
    return np.zeros_like(x)
    
xs = np.linspace(-5.0, 3.5, 100)
ys = f(xs)

'''
plt.figure()
plt.title('The hidden function f(x)')               
plt.plot(xs, ys)
plt.xlabel('x')
plt.ylabel('f(x)')
'''

x_obs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
y_obs = f(x_obs)

x_s = np.linspace(-8, 7, 80)
K = k(x_obs, x_obs)
K_s = k(x_obs, x_s)
K_ss = k(x_s, x_s)

K_sTKinv = np.matmul(K_s.T, np.linalg.pinv(K))
mu_s = m(x_s) + np.matmul(K_sTKinv, y_obs - m(x_obs))
Sigma_s = K_ss - np.matmul(K_sTKinv, K_s)
plt.figure()
y_true = f(x_s)
plt.plot(x_s, y_true, 'k--', linewidth=3, label='True f(x)')
plt.plot(x_obs, y_obs,'+', linewidth=30, label='Training data')


stds = np.sqrt(Sigma_s.diagonal())
err_xs = np.concatenate((x_s, np.flip(x_s, 0)))
err_ys = np.concatenate((mu_s + 2 * stds, np.flip(mu_s - 2 * stds, 0)))
plt.fill(err_xs, err_ys,  alpha=.5, fc='grey', ec='None',label='Uncertainty')

y_s = np.random.multivariate_normal(mu_s, Sigma_s)
plt.plot(x_s, y_s, 'r-', label='MEAN')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-7, 8)
label = ['True f(x)','Training data', 'Uncertainty','MEAN']
plt.legend(label, loc = 0, ncol = 1)

plt.show()