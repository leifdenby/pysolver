


class ForwardEuler1D:
    def __init__(self, fluxMethod):
        self.fluxMethod = fluxMethod

    def __call__(self, Q, dx, dt):
        num_ghost_cells = self.fluxMethod.num_ghost_cells

        f = self.fluxMethod(Q, dx, dt)

        Q[num_ghost_cells:-num_ghost_cells] += dt/dx*(f[:-1] - f[1:])
        
    def __str__(self):
        return "ForwardEuler1D (%s)" % self.fluxMethod


class ForwardEuler2D:
    def __init__(self, fluxMethod, model):
        self.FG = fluxMethod(model)
        self.fluxMethod = fluxMethod
        if hasattr(self.FG, "maxCFL"):
            self.maxCFL = self.FG.maxCFL

    def __call__(self, Q, dx, dy, dt):

        (f, g) = self.FG(Q, dx, dy, dt)

        n = self.fluxMethod.num_ghost_cells
        # flux between ghost cells in normal direction to flux are also calculated, so these must be discarded
        Q[n:-n,n:-n] += dt/dx*(f[:-1,n:-n] - f[1:,n:-n]) + dt/dy*(g[n:-n,:-1] - g[n:-n,1:])

    def __str__(self):
        return "ForwardEuler2D (%s)" % self.fluxMethod
