"""
This module enables us to create and apply boundary conditions in a generic
way, we may use the 'primitive' boundary conditions (Dirichlet, Neumann or
Periodic) and create new derived BCs from these, e.g. the MovingWallBC

These may then be applied with applyBCs (see test() as an example)
"""


class Neumann(object):
    """
        General class for describing a Neumann boundary condition
    """
    def __init__(self, slope):
        self.slope = slope

class Dirichlet(object):
    """
        General class for describing a Dirichlet boundary condition.
    """
    def __init__(self, fixed_value):
        self.fixed_value = fixed_value

class Periodic(object):
    pass

class ZeroFlux(Dirichlet):
    def __init__(self):
        super(ZeroFlux, self).__init__(fixed_value=0.0)

class ZeroGradient(Neumann):
    def __init__(self):
        super(ZeroGradient, self).__init__(slope=0.0)

class FixedGradient(Neumann):
    def __init__(self, gradient):
        super(FixedGradient, self).__init__(slope=gradient)

class Outflow(Dirichlet):
    def __init__(self, velocity):
        super(Outflow, self).__init__(fixed_value=velocity)

class Transmissive(ZeroGradient):
    pass

class NoSlip(ZeroFlux):
    pass

class SolidWall(ZeroFlux):
    pass

class MovingWall(Dirichlet):
    def __init__(self, wall_velocity):
        self.wall_velocity = wall_velocity
        super(MovingWall, self).__init__(fixed_value=wall_velocity)

class Adiabatic(ZeroGradient):
    pass

class Isothermal(Dirichlet):
    def __init__(self, temp):
        super(Isothermal, self).__init__(fixed_value=temp)

def test():
    from grids.boundary_condition_helper import applyCellCenteredBCs
    import numpy as np
    from grids import grid2d
    num_ghost_cells = 2
    N = 5
    grid = grid2d.FV(N = (N, N), num_ghost_cells=num_ghost_cells, x=(0.0, 1.0), y=(0.0, 1.0))

    Q = np.random.randint(10, size=(N + 2*num_ghost_cells,N + 2*num_ghost_cells, 2))

    from common import print_grid
    print_grid(Q[...,1])

    boundary_conditions = { 0: (SolidWall(), SolidWall(), NoSlip(), NoSlip()), 
                            1: (MovingWall(wall_velocity=1.0), NoSlip(), NoSlip(), NoSlip()) }

    applyCellCenteredBCs(Q=Q, all_boundary_conditions=boundary_conditions, num_ghost_cells=num_ghost_cells, grid=grid)

    print_grid(Q[...,1])

if __name__ == "__main__":
    test()

