# A simple python module implementing SDE solvers

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def euler_maruyama(drift, diffusion, x_0, T, N = 100):
    """
    Algorithm for Euler-Maruyama method
    Solves SDE of form dX_t = drift(X_t)dt + diffusion(X_t)dW_t, X_0 = x_0 in the interval [0,T]
    N = number of subintervals of [0, T] used during computation
    Returns the computed path
    """
    del_t = T/float(N)
    root_t = np.sqrt(del_t)
    # figure out dimension of the problem
    if not np.isscalar(x_0):
        dimension = len(x_0)
        Y = np.zeros((N+1, dimension))
        mult = np.matmul
    else:
        dimension = None
        Y = np.zeros(N+1)
        mult = lambda p, q: p*q
    # initialize and recurse
    Y[0] = x_0
    for i in range(1, N+1):
        Y[i] = Y[i-1] + drift(Y[i-1])*del_t + mult(diffusion(Y[i-1]), np.random.normal(0.0, root_t, size = dimension))
    return Y

class SDE(object):
    """
    This is a class for defining an SDE of form dX_t = drift(X_t)dt + diffusion(X_t)dW_t, X_0 = x_0
    """
    def __init__(self, drift, diffusion, x_0):
        self.drift = drift
        self.diffusion = diffusion
        self.x_0 = x_0
        if not np.isscalar(x_0):
            self.dim = len(x_0)
        else:
            self.dim = 1

    def solve(self, method, T, N = 1000):
        """
        Numerically solves sde in [0, T] using the preferred method
        N = number of subintervals of [0, T] used during computation
        """
        if method == "euler_maruyama":
            self.Y = euler_maruyama(self.drift, self.diffusion, self.x_0, T, N)
        self.T = T

    def expected_path(self, num_paths = 1000, method, T, N = 1000):
        for i in range(num_paths):
            self.

    def draw_path_1(self):
        """
        Draws the generated path (against time) for 1-dimensional problems
        """
        x = np.linspace(0.0, self.T, len(self.Y), endpoint = True)
        plt.plot(x, self.Y)
        plt.show()

    def draw_path_3(self, max_pts = 100):
        """
        Draws the generated trajectory for 3-dimensional problems
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        pts = len(self.Y[:, 0])
        # select points to plot
        if pts > max_pts:
            increment = int(pts/max_pts)
            x, y, z = np.zeros(max_pts), np.zeros(max_pts), np.zeros(max_pts)
            j = 0
            for i in range(max_pts):
                x[i] = self.Y[j, 0]
                y[i] = self.Y[j, 1]
                z[i] = self.Y[j, 2]
                j += increment
        else:
            x, y, z = self.Y[:, 0], self.Y[:, 1], self.Y[:, 2]
        ax.plot3D(x,y,z)
        plt.show()

"""
# For testing .....
if __name__ == '__main__':
    a = lambda x: 0.5*x
    b = lambda x: x
    x_0 = 1.0
    sde = SDE(a, b, x_0)
    sde.solve(method = 'euler_maruyama', T = 10, N=100200)
    sde.draw_path_1()
"""
