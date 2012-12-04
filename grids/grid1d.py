import numpy as np
import common

class FVGrid1D:
    def __init__(self, N, num_ghost_cells, boundary_conditions, x):
        (self.Nx,) = N
        (self.x0, self.x1) = x
        (self.dx,) = self.getGridSpacing()
        self.num_ghost_cells = num_ghost_cells
        self.boundary_conditions = boundary_conditions

    def initiateCells(self, ic_func, model):
        (xs,) = self.getCellCenterPositions()

        Q = np.zeros((self.Nx+2*self.num_ghost_cells, model.state.num_components))
        for i in range(self.Nx):
            Q[i+self.num_ghost_cells] = ic_func(xs[i])

        self.applyBoundaryConditions(Q)
        return Q

    def getCellCenterPositions(self):
        x = np.linspace(self.x0 + self.dx*0.5, self.x1 - self.dx*0.5, self.Nx)
        return (x, )

    def getGridSpacing(self):
        return ( (self.x1-self.x0)/self.Nx, )

    def getExtent(self):
        return (self.x0, self.x1, )

    def applyBoundaryConditions(self, Q):
        if self.boundary_conditions == (BoundaryCondition.PERIODIC, ):
            Q[-1] = Q[1]
            Q[0] = Q[-2]
        elif self.boundary_conditions == (BoundaryCondition.TRANSMISSIVE, ):
            Q[0] = Q[1]
            Q[-1] = Q[-2]
        else:
            raise Exception("BoundaryCondition not implemented")

    def getDomainWidthAndCenter(self):
        return (self.x1-self.x0, (self.x0+self.x1)/2.0)
