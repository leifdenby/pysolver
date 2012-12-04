


class GravitySourceTerm2D:
    """
        First attempt at implementing a first order gravity source term integrator
    """
    def __init__(self, model, g = 1.0):
        self.model = model
        self.g = g

    def __call__(self, Q, dx, dy, dt, num_ghost_cells):
        n = num_ghost_cells
        vec = Q.view(self.model.state)
        vec.y_velocity[n:-n,n:-n] += -self.g*dt

