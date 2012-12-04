import numpy as np
from scipy.sparse import csc_matrix

import common
from grids import boundary_conditions as BCs

from scipy.sparse import spdiags, bmat, coo_matrix, eye as speye, kron as sp_kron
from scipy.sparse.linalg import spsolve
from scipy.constants import epsilon_0, e as e_charge, pi
from numpy import meshgrid, linspace, vectorize, ones, shape, resize
import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d

import scipy.sparse
import pyamg



def main():
    #np.set_printoptions(precision=4)
    #operator_test(N=3)
    #import cProfile
    #cProfile.run("test1(N=500)")
    #graphic_of_discrete_operator(4)
    #test_result_plot(test2)
    convergence_test(test1,N_max=80)

def test_result_plot(test_func):

    N = 30
    (u, u_exact) = test_func(N)
    
    plot.ion()
    figure=plot.figure()
    ax = mplot3d.Axes3D(figure)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    x_, y_ = common.meshgrid(linspace(0.0, 2*pi, N), linspace(0.0, 2*pi, N))
    #ax.plot_wireframe(x_, y_, u_exact, label = "N= %i" % N, color="red")
    #ax.plot_wireframe(x_, y_, u, label = "N= %i" % N)
    #ax.set_zlim3d((np.mean(u-u_exact)-1.0, np.mean(u-u_exact) + 1.0))
    

    plot.imshow(np.rot90(u-u_exact), extent=[0.0, 2.0*pi, 0.0, 2.0*pi])
    plot.colorbar()
    plot.xlabel("x")
    plot.ylabel("y")

    plot.draw()
    raw_input()




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
    
    K2D = discrete_laplacian_2d(N, N, (BCs.Neumann, BCs.Neumann, BCs.Neumann, BCs.Neumann))

    u = spsolve(K2D, b)

    return (x_, y_, resize(u, (N,N)) )

def test3(N):
    """
        Test-case for solving the non-linear Poisson equation of the form
            div(a(x,y)grad(phi)) = b(x,y),
        using Dirichlet boundary conditions on all boundaries, HOWEVER with a=1, so that effectively
        we are solving a normal Poisson equation.

        We pick the solution phi(x,y) = sin(x)*sin(y) so that phi = 0 on the boundary, the Dirichlet BC is zero and
        so does not need to be included in the source term.
    """
    from numpy import cos, sin

    # grid setup
    ((x_min, x_max), (y_min, y_max)) = ((0.0, 2*pi), (0.0, 2*pi))
    dx = (x_max - x_min)/N
    dy = (y_max - y_min)/N
    x = linspace(x_min, x_max, N)
    y = linspace(y_min, y_max, N)
    xx = np.zeros((N, N))
    yy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            xx[i,j] = x[i]
            yy[i,j] = y[j]
    
    # test where a=1, so the Poisson equation is homogeneous
    a_x = np.ones((N+1, N))/dx**2.0
    a_y = np.ones((N, N+1))/dy**2.0
    a = (a_x, a_y)


    def u_f(x,y):
        return sin(xx)*sin(yy)


    u_exact = u_f(xx,yy)
    b = -2*u_f(xx,yy)
    
    grid = (N, N)
    bcs = (BoundaryCondition.DIRICHLET, BoundaryCondition.DIRICHLET, BoundaryCondition.DIRICHLET, BoundaryCondition.DIRICHLET)

    K2D = discrete_nonlinear_laplacian_2d(grid, a, bcs)

    u = spsolve(K2D, b.ravel())
    u = u.reshape((N, N))

    # the corner is inforced by the discrete Laplacian to be equal to 0.0, however this is not what we know that the exact value will be here
    # so we add the value of the exact solution at the corner, and offset the solution. This is valid because with only Neumann BCs the solution
    # is unique only up to an additive constant (as only derrivatives are defined on the boundary).
    #u = u + u_exact[0][0] 


    return (u, u_exact)

def test1(N):
    """
    Test for the homogeneous discrete Laplacian Poisson solver.

    We subtract pi because the discrete Laplacian is made so it gaurantees the bottom left corner to be 0.
    
    Short rutine for plotting one run. 
        N = 50
        (u, u_exact) = test1(N)
        
        plot.ion()
        figure=plot.figure()
        plot.imshow(u-u_exact, extent=[0.0, 2.0*pi, 0.0, 2.0*pi])
        plot.colorbar()

        plot.draw()
        raw_input()
    """

    from numpy import cos

    # grid setup
    ((x_min, x_max), (y_min, y_max)) = ((0.0, 2*pi), (0.0, 2*pi))
    dx = (x_max - x_min)/N
    dy = (y_max - y_min)/N
    x = linspace(x_min, x_max, N)
    y = linspace(y_min, y_max, N)
    xx = np.zeros((N, N))
    yy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            xx[i,j] = x[i]
            yy[i,j] = y[j]
    
    # test where a=1, so the Poisson equation is homogeneous
    a_x = np.ones((N+1, N))#/dx**2.0
    a_y = np.ones((N, N+1))#/dy**2.0
    a = (a_x, a_y)


    # we assumed a solution p(x,y) = cos(x)*cos(y), so that dp/dn = 0
    u_exact = cos(xx)*cos(yy)


    b = -2*cos(xx)*cos(yy)
    
    grid = (N, N)
    bcs = (BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN)

    K2D = new_discrete_nonlinear_laplacian_2d(grid, a, bcs)
    #K2D_old = discrete_nonlinear_laplacian_2d(grid, a, bcs)

    #print K2D_old.todense()
    #print
    #print K2D.todense()

    #raw_input()

    u = scipy.sparse.linalg.spsolve(K2D, b.ravel())
    u = u.reshape((N, N))

    # the corner is inforced by the discrete Laplacian to be equal to 0.0, however this is not what we know that the exact value will be here
    # so we add the value of the exact solution at the corner, and offset the solution. This is valid because with only Neumann BCs the solution
    # is unique only up to an additive constant (as only derrivatives are defined on the boundary).
    u = u + (u_exact[0][0] - u[0][0])


    return (u, u_exact)

def test2(N, amg=True):
    """
        Test-case for solving the non-linear Poisson equation of the form
            div(a(x,y)grad(phi)) = b(x,y),
        using Neumann boundary conditions on all boundaries.

        We pick the solution phi(x,y) = cos(x)*cos(y) so that d(phi)/dn = 0, the Nuemann BC is zero and
        so does not need to be included in the source term.
    """
    from numpy import cos, sin
    # inhomogenous term, with derivatives
    t = 1.0
    def a_func(x,y):
        return 2.0 + sin(t*x)*sin(y)

    def dadx_func(x,y):
        return t*cos(t*x)*sin(y)

    def dady_func(x,y):
        return sin(t*x)*cos(y)

    # grid setup, for evaluating the source term
    ((x_min, x_max), (y_min, y_max)) = ((0.0, 2*pi), (0.0, 2*pi))
    dx = (x_max - x_min)/N
    dy = (y_max - y_min)/N

    x = linspace(x_min+dx/2.0, x_max-dx/2.0, N)
    y = linspace(y_min+dy/2.0, y_max-dy/2.0, N)
    xx = np.zeros((N, N))
    yy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            xx[i,j] = x[i]
            yy[i,j] = y[j]
    
    # grid positions including ghost cells (for evaluating the inhomogeneous term)
    import common
    x_ = linspace(x_min, x_max, N+1)
    y_ = linspace(y_min+dy/2.0, y_max-dy/2.0, N)
    xx_, yy_ = common.meshgrid(x_,y_)
    a_x = a_func(xx_, yy_)
    
    x_ = linspace(x_min+dx/2.0, x_max-dx/2.0, N)
    y_ = linspace(y_min, y_max, N+1)
    xx_, yy_ = common.meshgrid(x_,y_)
    a_y = a_func(xx_, yy_)


    (N_x, N_y) = N, N
    a = (a_x, a_y)

    # we've assumed a solution p(x,y) = cos(x)*cos(y), so that dp/dn = 0
    u_exact = cos(xx)*cos(yy)

    b = -(dadx_func(xx,yy)*sin(xx)*cos(yy) + 2*a_func(xx,yy)*cos(xx)*cos(yy) + dady_func(xx,yy)*cos(xx)*sin(yy))
    
    grid = (N, N)
    bcs = (BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN)

    #K2D = new_discrete_nonlinear_laplacian_2d(grid, a, bcs)
    K2D = discrete_nonlinear_laplacian_2d(grid, a, bcs)


    #if amg:
        #ml = pyamg.ruge_stuben_solver(K2D)
        #u = ml.solve(b.ravel(), tol=1.0e-10)
    #else:
        #import scipy.sparse.linalg
        #u = spsolve(K2D, b.ravel())
        #(u, info) = scipy.sparse.linalg.cg(K2D, b.ravel(), tol=1.0e-10)
    #u = u.reshape((N, N))

    u = u_exact

    # the corner is inforced by the discrete Laplacian to be equal to 0.0, however this is not what we know that the exact value will be here
    # so we add the value of the exact solution at the corner, and offset the solution. This is valid because with only Neumann BCs the solution
    # is unique only up to an additive constant (as only derrivatives are defined on the boundary).
    u = u + u_exact[0][0] - u[0][0] 

    return (u, u_exact)


def graphic_of_discrete_operator(N):
    # test where a=1, so the Poisson equation is homogeneous
    a_x = np.ones((N+1, N))
    a_y = np.ones((N, N+1))
    a = (a_x, a_y)

    grid = (N, N)
    bcs = (BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN)
    K2D = discrete_nonlinear_laplacian_2d(grid, a, bcs)


    data = np.abs(np.array(K2D.todense()))

    import matplotlib.cm as cm
    plot.ion()
    plot.imshow(data, interpolation="nearest", cmap=cm.Greys)
    plot.xlabel("j")
    plot.ylabel("i")
    plot.draw()
    #plot.savefig("K2D_%ix%i.png" % (N, N), dpi=400)
    raw_input()


def convergence_test(test_func, N_max=80, norm=2, save_png=False):
    results = []
    test_name = test_func.__name__
    print("Running %s" % test_name)
    
    for N in range(10,N_max,2):
        print "N=%i" % N

        (u, u_exact) = test_func(N)

        norm_err = np.linalg.norm(u-u_exact, ord=norm)
        
        results.append((N, norm_err/(N*N)))


    results = np.array(results)

    N = results[:,0]
    dx = 1.0/N
    err = results[:,1]

    (a, b) = np.polyfit(np.log(dx), np.log(err), 1)

    print "slope=%.3f" % a

    plot.ion()
    plot.plot(np.log(dx), np.log(err), '*', label="$L_{%s}$ norm" % str(norm))
    plot.plot(np.log(dx), b + a*np.log(dx), label="fit (slope=%.3f)" % a)
    plot.xlabel("ln(dx)")
    plot.ylabel("ln(err)")
    plot.legend(loc=4)
    plot.grid(True)
    
    if save_png:
        plot.savefig("nonlinear_poisson_convergence_%s.png" % (test_name), dpi=400)

    plot.draw()
    raw_input()

def operator_test(N):

    a_x = np.floor(1.0+10.0*np.random.random((N+1,N)))
    a_y = np.floor(1.0+10.0*np.random.random((N,N+1)))
    bcs = (BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN, BoundaryCondition.NEUMANN)

    K2D = new_discrete_nonlinear_laplacian_2d((N, N), (a_x, a_y), bcs)
    K2D_old = discrete_nonlinear_laplacian_2d((N, N), (a_x, a_y), bcs)

    print "a_x"
    print a_x.T
    print
    print "a_y"
    print a_y.T
    print

    print K2D_old.todense()
    print
    print K2D.todense()

def new_discrete_nonlinear_laplacian_2d(N, a, boundaryConditions):
    (bc_W, bc_E, bc_S, bc_N) = boundaryConditions
    (Nx, Ny) = N
    (a_x, a_y) = a

    # check that a has the right dimensions
    if not a_x.shape == (Nx+1, Ny) or not a_y.shape == (Nx, Ny+1):
        raise Exception('The nonlinearizing term has the wrong shape, it is expected to be defined at cell interfaces')
    
    # TODO: Implementation below would be more efficient, we're using multiple
    # append-operations now
    # Create arrays to contain coeffients
    #N_nonzero = 5*N_x*N_y - 2*(N_x+N_y)
    #row = np.empty((N_nonzero,1))
    #column = np.empty((N_nonzero,1))
    #data = np.empty((N_nonzero,1))

    cN = -1 if isinstance(bc_N, BCs.Neumann) else 1
    cS = -1 if isinstance(bc_S, BCs.Neumann) else 1
    cE = -1 if isinstance(bc_E, BCs.Neumann) else 1
    cW = -1 if isinstance(bc_W, BCs.Neumann) else 1

    rows = []
    cols = []
    data = []

    # take each point of five-point stencil in turn
    # North
    i = np.arange(Nx*Nx-1)
    j = np.arange(1,Nx*Ny)
    k = i / Nx
    l = i % Nx
    rows = np.append(rows, i)
    cols = np.append(cols, j)
    data = np.append(data, (l!=Ny-1)*a_y[k,l+1])
    
    # South
    i = np.arange(1,Nx*Ny)
    j = np.arange(Nx*Nx-1)
    k = i / Nx
    l = i % Nx
    rows = np.append(rows, i)
    cols = np.append(cols, j)
    data = np.append(data, (l!=0)*a_y[k,l])
    
    # East
    i = np.arange(Nx*Nx-Nx)
    j = np.arange(Nx,Nx*Ny)
    k = i / Nx
    l = i % Nx
    rows = np.append(rows, i)
    cols = np.append(cols, j)
    data = np.append(data, (k!=Nx-1)*a_x[k+1,l])
    
    # West
    i = np.arange(Nx,Nx*Ny)
    j = np.arange(Nx*Nx-Nx)
    k = i / Nx
    l = i % Nx
    rows = np.append(rows, i)
    cols = np.append(cols, j)
    data = np.append(data, (k!=0)*a_x[k,l])

    # Center
    i = np.arange(Nx*Ny)
    j = np.arange(Nx*Nx)
    k = i / Nx
    l = i % Nx
    rows = np.append(rows, i)
    cols = np.append(cols, j)
    val = -(  
                (1.0 + (k==Nx-1)*cE)*a_x[k+1,l] 
                + (1.0 + (k==0)*cW)*a_x[k,l] 
                + (1.0 + (l==0)*cS)*a_y[k,l]  
                + (1.0 + (l==Ny-1)*cN)*a_y[k,l+1] 
            )
    h = a_x[k+1,l], a_x[k,l], a_y[k,l], a_y[k,l+1]
    data = np.append(data, val)

    from scipy.sparse import coo_matrix

    m = coo_matrix((data,(rows,cols)),shape=(Nx*Ny,Nx*Ny))
    return m.tocsr()

def discrete_nonlinear_laplacian_2d(N, a, boundaryConditions):
    """
    Creates a discrete form of the nonlinear 2D Laplacian for solving e.g. a nonlinear Poissions equation.
    The unknown p when solving the nonlinear equation 
        div(a(x,y) grad(p(x,y))) = D(a,p) = u(x,y), 
    will be a vector of length N_x*N_y and D will be (N_x*N_y)^2

    The term making the Laplacian nonlinear (a) is split into two contributions, a_x being the interface values in the
    x-direction, and a_y the interface values in the y-direction.
    a_x is thus expected to have (N_x+1) by (N_y) entries, whereas a_y is expected to be (N_x) by (N_y+1)
    
    N should be a tuple as N = (N_x, N_y)
    a should be a tuple as a = (a_x, a_y)
    Boundary conditions are defined in the order (W, E, S, N)

    NB: The discretization is assumed to be included by being scaled into a_x and a_y, and included in the source term
    when the discrete Poission equation is solved. It is assumed that the Poisson equation is being solved at cell centers 
    and so the boundary conditions will be satisfied at dx/2 from the solution point (this has implications for how the boundary
    conditions are implemented).

    The return matrix is in csc_matrix format of scipy.sparse
    """
    (bc_W, bc_E, bc_S, bc_N) = boundaryConditions
    (N_x, N_y) = N
    (a_x, a_y) = a

    # check that a has the right dimensions
    if not a_x.shape == (N_x+1, N_y) or not a_y.shape == (N_x, N_y+1):
        raise Exception('The nonlinearizing term has the wrong shape, it is expected to be defined at cell interfaces')
    
    # create a matrix with the right number of rows and columns
    p_length = N_x*N_y
    size_mb = p_length*p_length*32*2.0/8/1024/1024/1024
    if size_mb > 1.0:
        raise Exception("This is the old Laplacian-operator allocator, the operator you just requested would take up more than 1gb of memory. Allocation has been halted.")
    D = np.zeros((p_length, p_length))

    # set up stencil for centered differences of the nonlinear Laplacian operator
    # if we are at an edge cell we should apply the boundary conditions, and the neighbouring
    # cell in the direction of the edge should not be set (as it doesn't exist)

    # D represents the discrete Laplacian operator in its entire form. Row i*N_y+j is the row for which
    # cell (i,j) is center of the stencil
    for i in range(0,N_x):
        for j in range(0,N_y):
            # Calculate the interface values of the nonlinearizing term a(x,y)
            a_N = a_y[i,j+1]
            a_S = a_y[i,j]
            a_E = a_x[i+1,j]
            a_W = a_x[i,j]

            D[i*N_y+j,i*N_y+j] = -(a_N+a_S+a_E+a_W)
            # East
            if i+1 == N_x:
                if isinstance(bc_E, BCs.Neumann):
                    # first order, the center of the stencil of the current target cell is changed
                    D[i*N_y+j,i*N_y+j] += a_E
                    # second order
                    #D[i*N_y+j,(i-1)*N_y+j] += a_E
                elif isinstance(bc_E, BCs.Dirichlet):
                    D[i*N_y+j,i*N_y+j] -= a_E
                else:
                    raise Exception("Unknown boundary conditions type (%s)" % str(bc_E))
            else:
                # not on the boundary, so just add the stencil point like normal
                D[i*N_y+j,(i+1)*N_y+j] = a_E

            # W
            if i == 0:
                if isinstance(bc_W, BCs.Neumann):
                    D[i*N_y+j,i*N_y+j] += a_W
                    #D[i*N_y+j,(i+1)*N_y+j] += a_W
                #if j == 0:
                    #D[i*N_y+j,i*N_y+j] -= 1
                elif isinstance(bc_W, BCs.Dirichlet):
                    D[i*N_y+j,i*N_y+j] -= a_W
                else:
                    raise Exception("Unknown boundary conditions type (%s)" % str(bc_W))
            else:
                D[i*N_y+j,(i-1)*N_y+j] = a_W

            # N
            if j+1 == N_y:
                if isinstance(bc_N, BCs.Neumann):
                    D[i*N_y+j,i*N_y+j] += a_N
                elif isinstance(bc_N, BCs.Dirichlet):
                    D[i*N_y+j,i*N_y+j] -= a_N
                else:
                    raise Exception("Unknown boundary conditions type (%s)" % str(bc_N))
            else:
                D[i*N_y+j,i*N_y+(j+1)] = a_N

            # S
            if j == 0:
                if isinstance(bc_S, BCs.Neumann):
                    D[i*N_y+j,i*N_y+j] += a_S
                    #D[i*N_y+j,i*N_y+(j+1)] += a_S
                elif isinstance(bc_S, BCs.Dirichlet):
                    D[i*N_y+j,i*N_y+j] -= a_S
                else:
                    raise Exception("Unknown boundary conditions type (%s)" % str(bc_S))
            else:
                D[i*N_y+j,i*N_y+(j-1)] = a_S

    return csc_matrix(D)

def compare_new_to_old():
    N = 3
    a_x = np.floor(10.0*np.random.random((N+1,N)))
    a_y = np.floor(10.0*np.random.random((N, N+1)))
    common.print_grid(a_x)
    common.print_grid(a_y)
    bcs = (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())
    D_old = discrete_nonlinear_laplacian_2d((N, N), (a_x, a_y), boundaryConditions = bcs)
    D_new = new_discrete_nonlinear_laplacian_2d((N, N), (a_x, a_y), boundaryConditions = bcs)

    print "old"
    print D_old.todense()
    print "new"
    print D_new.todense()

    print np.equal(D_old.todense(), D_new.todense())
    
    common.print_grid(D_old - D_new)



def profile_func():
    for N in range(10,100,10):
        test3(N)

def profile_code():
    import cProfile
    cProfile.run("profile_func()")

if __name__ == "__main__":
    #main()
    compare_new_to_old()
