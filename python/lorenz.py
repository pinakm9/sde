import sde
import numpy as np

# define the lorenz SDE
sigma, rho, beta, lam = 10.0, 28.0, 8/3.0, 0.05

def a(X_t):
    x, y, z = X_t
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def b(X_t):
    x, y, z = X_t
    return lam*(x + y + z)*np.ones((3,3))

x_0 = [1.0, 1.0, 1.0]
lorenz = sde.SDE(a, b, x_0)
lorenz.solve(method = 'euler_maruyama', T = 10)
print (lorenz.Y)
lorenz.draw_path_3()
