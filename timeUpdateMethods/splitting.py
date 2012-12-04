import numpy as np


class FirstOrder2D:
    def __init__(self, flux_method, model):
        self.model = model
        self.F = flux_method(model.F)
        self.G = flux_method(model.G)
        self.fluxMethod = flux_method

    def __call__(self, Q, dx, dy, dt):
        n = self.fluxMethod.num_ghost_cells

        # solve for x-direction first
        f = self.F(Q, dx, dt)
        Q[n:-n,n:-n] += dt/dx*(f[:-1,n:-n] - f[1:,n:-n])
        
        # swap axes and solve for the y-direction using solution from above as initial condition
        Q = np.swapaxes(Q, 0, 1)
        g = self.G(Q, dy, dt)
        Q[n:-n,n:-n] += dt/dy*(g[:-1,n:-n] - g[1:,n:-n])
        
        # swap back axes
        Q = np.swapaxes(Q, 0, 1)

    def __str__(self):
        return "FirstOrderSplit2D (%s)" % self.fluxMethod


class SecondOrder2D:
    """
        Second order splitting for two-dimensional hyperbolic problem. This is the so-called
        Strang splitting approach to second order splitting. Toro p. 546
    """
    def __init__(self, flux_method, model):
        self.model = model
        self.F = flux_method(model.F, model.state)
        self.G = flux_method(model.G, model.state)
        self.flux_method = flux_method

    def __call__(self, Q, dx, dy, dt):
        n = self.flux_method.num_ghost_cells

        # solve for x-direction first, dt/2.0
        f = self.F(Q, dx, dt/2.0)
        Q[n:-n,n:-n] += dt/(2.0*dx)*(f[:-1,n:-n] - f[1:,n:-n])
        
        # swap axes and solve for the y-direction using solution from above as initial condition
        Q = np.swapaxes(Q, 0, 1)
        g = self.G(Q, dy, dt)
        Q[n:-n,n:-n] += dt/dy*(g[:-1,n:-n] - g[1:,n:-n])
        
        # swap back axes
        Q = np.swapaxes(Q, 0, 1)
        # and solve for the last half timestep in x
        f = self.F(Q, dx, dt/2.0)
        Q[n:-n,n:-n] += dt/(2.0*dx)*(f[:-1,n:-n] - f[1:,n:-n])


    def __str__(self):
        return "SecondOrderSplit2D (%s)" % self.flux_method
