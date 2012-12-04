import numpy as np
import common

from boundary_condition_helper import applyCellCenteredBCs

class Edges:
    W = (-1,0)
    E = (1,0)
    S = (0,-1)
    N = (0,1)

edges = [Edges.W, Edges.E, Edges.S, Edges.N]

class StatePosition:
    center = 0
    interface_x = 1
    interface_y = 2
    corner = 3

class DomainSpec:
    def __init__(self, N, x, y):
        (self.Nx, self.Ny) = N
        self.N = N
        (self.x0, self.x1) = x
        (self.y0, self.y1) = y
        (self.dx, self.dy) = self.getGridSpacing()
        self.edges = edges
    
    def getGridSpacing(self):
        return ( (self.x1-self.x0)/self.Nx, (self.y1-self.y0)/self.Ny)

    def getExtent(self, n = 0):
        return (self.x0 - n*self.dx, self.x1 + n*self.dx, self.y0 - n*self.dy, self.y1 + n*self.dy)

    def getLengthScales(self):
        extent = self.getExtent()
        return [abs(extent[0]-extent[1]), abs(extent[2]-extent[3])]

    def getNumCells(self, n = 0):
        return (self.Nx + n*2, self.Ny + n*2)

    def getCenter(self):
        return ((self.x1-self.x0)/2.0, (self.y1-self.y0)/2.0)



class FV(DomainSpec):
    def __init__(self, domain_spec, num_ghost_cells):
        self.__dict__.update(domain_spec.__dict__)
        self.num_ghost_cells = num_ghost_cells

    def initiateCells(self, test, model):
        (xs, ys) = self.getCellCenterPositions()

        Q = self.getEmptyGrid(model)
        for i in range(self.Nx):
            for j in range(self.Ny):
                pos = np.array([xs[i,j], ys[i,j]])
                Q[i+self.num_ghost_cells,j+self.num_ghost_cells] = np.array([f(pos) for f in test.ic_func])
        return Q

    def getEmptyGrid(self, model):
        return np.zeros((self.Nx+2*self.num_ghost_cells, self.Ny+2*self.num_ghost_cells, model.state.num_components))

    def getCellCenterPositions(self, n = 0):
        x_l = self.x0 + self.dx*0.5 - self.dx*n
        x_r = self.x1 - self.dx*0.5 + self.dx*n
        y_b = self.y0 + self.dy*0.5 - self.dy*n
        y_t = self.y1 - self.dy*0.5 + self.dy*n
        x = np.linspace(x_l, x_r, self.Nx + 2*n)
        y = np.linspace(y_b, y_t, self.Ny + 2*n)
        return common.meshgrid(x, y)
    
    def getInterfacePositions(self, axis, num_ghost_cells = 0):
        """
            Get positions for cell interfaces, positions will be centered on the cell faces

            axis is expected to by either 0 or 1 depending on whether the interfaces in the
            x- or the y-direction are requested.
        """
        n = num_ghost_cells
        nx = axis
        ny = axis-1
        x_l = self.x0 + self.dx*0.5*(axis) - self.dx*n
        x_r = self.x1 - self.dx*0.5*(axis) + self.dx*n
        y_b = self.y0 - self.dy*0.5*(axis-1) - self.dy*n
        y_t = self.y1 + self.dy*0.5*(axis-1) + self.dy*n
        x = np.linspace(x_l, x_r, self.Nx + 2*n + (1-axis))
        y = np.linspace(y_b, y_t, self.Ny + 2*n + axis)
        return common.meshgrid(x, y)

    def makeInterfaceStates(self, Q, axis, boundary_conditions):
        """
        Returns the states at cell interfaces in the direction of axis
        """
        from common import aslice
        if boundary_conditions is not None:
            applyCellCenteredBCs(Q, boundary_conditions, self, self.num_ghost_cells)
        else:
            import warnings
            warnings.warn("No boundary_conditions passed to makeInterfaceStates, interface states on boundary may be wrong")

        return 0.5*(Q[aslice(axis,0,-1)] + Q[aslice(axis,1,None)])

    def applyBCs(self, Q, boundary_conditions, num_ghost_cells = None):
        applyCellCenteredBCs(Q, boundary_conditions, self, num_ghost_cells)
        
    def sliceFromEdge(self, edge_i, row_num, num_ghost_cells=0):
        """
        returns a slice for indexing a single row in the interior of the domain, useful
        when applying the pressure boundary condition in low Mach number projection
        methods. First layer internally is for row_num=0, second layer is num_row=1, first
        ghost cell layer will be num_row=-1
        """
        edge = edges[edge_i]
        def processIndex(n):
            if n == 0:
                return slice(None)
            elif n == -1:
                return num_ghost_cells + row_num
            elif n == 1:
                return -num_ghost_cells - row_num - 1
            else:
                raise Exception("Edge defintion was not -1, 0 or 1, edge definition wrong")
                
        return map(processIndex, edge)

class MAC(DomainSpec):
    def __init__(self, domain_spec, num_ghost_cells):
        self.__dict__.update(domain_spec.__dict__)
        self.num_ghost_cells = num_ghost_cells
        
    def initiateCells(self, test, model, positioning):
        return np.array([test.ic_func[i](self.getStatePositions(positioning[i], self.num_ghost_cells)) for i in range(model.state.num_components)])

    def getStatePositions(self, state_position, num_ghost_cells = 0):
        n = num_ghost_cells
        num_points_x = self.Nx + n*2
        num_points_y = self.Ny + n*2
        x_l = self.x0 + self.dx*0.5 - self.dx*n
        x_r = self.x1 - self.dx*0.5 + self.dx*n
        y_b = self.y0 + self.dy*0.5 - self.dy*n
        y_t = self.y1 - self.dy*0.5 + self.dy*n
        
        if state_position == StatePosition.interface_x:
            num_points_x -= 1
            x_l -= self.dx*0.5
            x_r += self.dx*0.5
        elif state_position == StatePosition.interface_y:
            num_points_y -= 1
            y_b -= self.dy*0.5
            y_t += self.dy*0.5
        elif state_position == StatePosition.corner:
            num_points_x -= 1
            num_points_y -= 1
            x_l -= self.dx*0.5
            x_r += self.dx*0.5
            y_b -= self.dy*0.5
            y_t += self.dy*0.5

        x = np.linspace(x_l, x_r, num_points_x)
        y = np.linspace(y_b, y_t, num_points_y)
        return common.meshgrid(x, y)
    
    def makeCellCenteredStates(self, Q, axis, boundary_conditions=None):
        """
        Returns the states at cell centers in the direction of axis
        """
        import warnings
        warnings.warn("makeCellCenteredStates does not apply BCs at the moment, does it need to?")

        return 0.5*(Q[common.aslice(axis,0,-1)] + Q[common.aslice(axis,1,None)])
    
