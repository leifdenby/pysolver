import numpy as np
from scipy.sparse import csc_matrix

from scipy.sparse import spdiags, bmat, coo_matrix, eye as speye, kron as sp_kron
from scipy.sparse.linalg import spsolve
from scipy.constants import epsilon_0, e as e_charge
from numpy import meshgrid, linspace, vectorize, ones, shape, resize
import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d

import grids.boundary_conditions as BCs


def main():
    p2_gradient()

def test1():
    domain_min = 0.0
    domain_max = 1.0

    test_id = 0

    def bc_func1d(x):
        if x == domain_min:
            return 1.0
        elif x == domain_max:
            return 1.0
        else:
            return 0.0

    def bc_func2d(x,y,dx):
        if x == domain_max or x == domain_min:
            return 0*dx
        if y == domain_max or y == domain_min:
            return 0*dx
        else:
            return 0.0

    def f_func2d(x,y):
        """ the source function """
        if abs(x-0.5) < 0.2 and abs(y-0.5) < 0.2:
            if x > 0.5:
                return 100.0
            else:
                return -100.0
        else:
            return 0.0

    figure=plot.figure()
    ax = mplot3d.Axes3D(figure)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for N in [50]: #range(100,1000,100):
        print "N=%i" % N
        (x_, y_, u) = poisson_2d(bc_func2d, f_func2d, domain_min, domain_max, N)

        ax.plot_wireframe(x_, y_, u, label = "N= %i" % N)

    plot.show()

def poisson_2d(bc_func, f_func, domain_min, domain_max, num_cells = 100):
    """ try to solve 1D Possion's equation on a grid 
    du/d^2x + du/d^2y = phi
    solution u and bc/source function values are assumed to be collocated
    
    """

    N = num_cells + 1# number of grid points, including boundary cells
    domain_len = domain_max-domain_min
    dx = domain_len/num_cells

    # array of grid point positions for function evaluation

    (x_, y_) = meshgrid(linspace(domain_min, domain_max, N), linspace(domain_min, domain_max, N))
    xx_ = x_.flatten()
    yy_ = y_.flatten()

    # setup b-"vector" (N by N here) with forcing and BCs (two extra cells are boundary cells)
    b = - vectorize(f_func)(xx_,yy_)*(dx**2.0) - vectorize(bc_func)(xx_, yy_, dx)
    print b
    
    K2D = discrete_laplacian_2d(N, N, (BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN))

    u = spsolve(K2D, b)

    return (x_, y_, resize(u, (N,N)) )

def p2_gradient():
    N = 10
    dx = 1.0/N

    s = np.zeros((N,N))

    s[0] = dx*4.0*np.ones((N,))
    s[-1] = -dx*4.0*np.ones((N,))

    print s

    s = s.reshape((N*N,))
    
    K2D = discrete_laplacian_2d(N, N, (BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN, BoundaryCondition_Pressure.NEUMANN))
    
    u = spsolve(K2D, s)
    u = u.reshape((N,N))
    
    x, y = meshgrid(linspace(0.0,1.0,N), linspace(0.0, 1.0, N))


    figure=plot.figure()
    ax = mplot3d.Axes3D(figure)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot_wireframe(x, y, u, label = "N= %i" % N)
    plot.show()


def discrete_laplacian_2d(N, M, BoundaryCondition_Pressures):
    """
    Creates a discrete form of the 2D Laplacian for solving e.g. Poissions equation.
    The unknown p when solving the linear equation Dp = u, will be a vector of length NxM
    and D will be (NxM)^2

    Boundary conditions are defined in the order (E, W, S, N)

    The return matrix is in csc_matrix format of scipy.sparse
    """
    (bc_E, bc_W, bc_S, bc_N) = BoundaryCondition_Pressures
    
    # create a matrix with the right number of rows and columns
    p_length = N*M
    D = np.zeros((p_length, p_length))

    # set up stencil for centered differences of D
    # if we are at an edge cell we should apply the boundary conditions, and the neighbouring
    # cell in the direction of the edge should not be set (as it doesn't exist)
    for i in range(0,N):
        for j in range(0,M):
            D[i*M+j,i*M+j] = -4
            # East
            if i+1 == N:
                if isinstance(bc_E, BCs.Neumann):
                    D[i*M+j,i*M+j] += 1
            else:
                D[i*M+j,(i+1)*M+j] = 1
            # W
            if i == 0:
                if isinstance(bc_W, BCs.Neumann):
                    D[i*M+j,i*M+j] += 1
            else:
                D[i*M+j,(i-1)*M+j] = 1
            # N
            if j+1 == M:
                if isinstance(bc_N, BCs.Neumann):
                    D[i*M+j,i*M+j] += 1
            else:
                D[i*M+j,i*M+(j+1)] = 1
            # S
            if j == 0:
                if isinstance(bc_S, BCs.Neumann):
                    D[i*M+j,i*M+j] += 1
            else:
                D[i*M+j,i*M+(j-1)] = 1
    
    return csc_matrix(D)

if __name__ == "__main__":
    main()
