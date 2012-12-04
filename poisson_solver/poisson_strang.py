"""
Learning how to code a Poisson solver, 1D and 2D written so far

Notes:
- The source function should probably by defined in a better way, since the volume of source material
    needs to be constant as the resolution is changed, this probably means passing the cell size to the
    source function
- It may be a the a more general Poisson solver doesn't take in a boundary condition function and a source
    function, and instead just a grid of the right dimensions, together with the cellspacing. This would make
    sense in respect to the solver I will eventually need.
- Currently the 2D solver assumes a grid ratio of unity, this should be changed.
- It would be good to implement the FFT Poisson solver too
"""

from scipy.sparse import spdiags, bmat, coo_matrix, eye as speye, kron as sp_kron
from scipy.sparse.linalg import spsolve
from scipy.constants import epsilon_0, e as e_charge
from numpy import meshgrid, linspace, vectorize, ones, shape, resize
import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d


def main():
    test2()
    quit()
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

    def f_func1d(x):
        return 0.0
        """ the source function """
        if x > 0.25 and x < 0.5:
            return 1.0
        elif x > 0.5 and x < 0.75:
            return -1.0
        else:
            return 0.0
    
    def bc_func2d(x,y):
        if test_id == 0:
            if x == domain_max or x == domain_min:
                return 1.0
            if y == domain_max or y == domain_min:
                return 1.0
            else:
                return 0.0
        elif test_id == 1:
            if x == domain_max and y == domain_max:
                return x+y
            if x == domain_max:
                return y
            if y == domain_max:
                return x
            else:
                return 0.0
        else:
            raise NotImplementedError("This test case hasn't been implemented yet")

    def f_func2d(x,y):
        return 0.0
        """ the source function """
        if abs(x-0.5) < 0.05 and abs(y-0.5) < 0.05:
            return -100.0
        else:
            return 0.0

    figure=plot.figure()
    ax = mplot3d.Axes3D(figure)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for N in [100]: #range(100,1000,100):
        print "N=%i" % N
        (x_, y_, u) = poisson_2d(bc_func2d, f_func2d, domain_min, domain_max, N)

        ax.plot_wireframe(x_, y_, u, label = "N= %i" % N)

    plot.show()

def test():
    N = 5# number of grid points, including boundary cells
    domain_len = 1.0
    dx = domain_len/(N-1)
    import numpy as np

    # setup b-vector with forcing and BCs (two extra cells are boundary cells)
    b = np.array([1.0*dx, 0, 0, 0, 0])
    
    # set up discrete differential operator matrix
    operator_coeffs = ones((3,N)) # three elements in this operator: 1, -2, 1
    operator_coeffs[1] = -2*operator_coeffs[1] # first column changed from 1 to -2

    diags = [-1, 0, 1] # diagonals to place these integers in
    B = spdiags(operator_coeffs, diags, N, N).tocsc()
    u = spsolve(B, b)

    print u
    print (u[1:]-u[:-1])/dx

def test2():
    N = 5# number of grid points, including boundary cells
    domain_len = 1.0
    dx = domain_len/(N-1)

    import numpy as np

    # setup b-vector with forcing and BCs (two extra cells are boundary cells)
    b = np.array([[1.0*dx, dx, dx, dx, dx], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0]] )
    b = b.reshape((25,))

    K2D = discrete_laplacian_2d(N)

    u = spsolve(K2D, b)

    from common import print_grid
    print_grid(u.reshape((5,5)))


def poisson_1d(bc_func, f_func, domain_min, domain_max, num_cells = 100):
    """ try to solve 1D Possion's equation on a grid 
    du/d^2x = phi
    solution u and bc/source function values are assumed to be collocated
    
    """

    N = num_cells + 1# number of grid points, including boundary cells
    domain_len = domain_max-domain_min
    dx = domain_len/num_cells

    # array of grid point positions for function evaluation
    x_ = linspace(domain_min, domain_max, N)

    # setup b-vector with forcing and BCs (two extra cells are boundary cells)
    b = - vectorize(f_func)(x_)*(dx**2.0) - vectorize(bc_func)(x_)
    
    # set up discrete differential operator matrix
    operator_coeffs = ones((3,N)) # three elements in this operator: 1, -2, 1
    operator_coeffs[1] = -2*operator_coeffs[1] # first column changed from 1 to -2

    diags = [-1, 0, 1] # diagonals to place these integers in
    B = spdiags(operator_coeffs, diags, N, N).tocsc()
    u = spsolve(B, b)

    return (x_, u, b, B)

def discrete_laplacian_2d(N):
    # set up discrete differential operator matrix
    operator_coeffs = (ones((N,3))*[1, -2, 1]).T # three elements in this operator: 1, -2, 1

    diags = [-1, 0, 1] # diagonals to place these integers in
    K1D = spdiags(operator_coeffs, diags, N, N)
    I1D = speye(N,N)
    K2D = sp_kron(K1D,I1D) + sp_kron(I1D,K1D)

    return K2D


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
    b = - vectorize(f_func)(xx_,yy_)*(dx**2.0) - vectorize(bc_func)(xx_,yy_)
    
    K2D = discrete_laplacian_2d(N)

    u = spsolve(K2D, b)

    return (x_, y_, resize(u, (N,N)) )



if __name__ == "__main__":
    main()
