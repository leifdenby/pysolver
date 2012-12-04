import numpy as np

USE_NUMEXPR = False
if USE_NUMEXPR:
    import numexpr as ne

epsilon = 1.0e-7

class Force2D:
    maxCFL = 0.332
    num_ghost_cells = 1

    def __init__(self, model):
        F = model.F
        G = model.G
        self.lf_instance_x = LaxFriedrichs1D(F, 2)
        self.lf_instance_y = LaxFriedrichs1D(G, 2)
        self.lw_instance_x = LaxWendroff1D(F)
        self.lw_instance_y = LaxWendroff1D(G)

    def __call__(self, Q, dx, dy, dt):
        n = Force2D.num_ghost_cells

        if USE_NUMEXPR:
            lf_x = self.lf_instance_x(Q, dx, dt)
            lw_x = self.lw_instance_x(Q, dx, dt)
            f = ne.evaluate("0.5*(lf_x + lw_x)")
            
            Q_y = np.swapaxes(Q, 0, 1)
            lf_y = self.lf_instance_y(Q_y, dy, dt)
            lw_y = self.lw_instance_y(Q_y, dy, dt)
            g = ne.evaluate("0.5*(lf_y + lw_y)")
            g_y = np.swapaxes(g, 0, 1)
           
            return (f[:,n:-n], g_y[n:-n,:])
        else:
            f = 0.5*(self.lf_instance_x(Q, dx, dt) + self.lw_instance_x(Q, dx, dt))
            Q_y = np.swapaxes(Q, 0, 1)
            g = 0.5*(self.lf_instance_y(Q_y, dy, dt) + self.lw_instance_y(Q_y, dy, dt))
            g_y = np.swapaxes(g, 0, 1)
            
            return (f[:,n:-n], g_y[n:-n,:])
    def __str__(self):
        return "Force2D"


class Force1D:
    num_ghost_cells = 1
    def __init__(self, model):
        self.lf_instance = LaxFriedrichs1D(model.F, 1)
        self.lw_instance = LaxWendroff1D(model.F)

    def __call__(self, Q, dx, dt):
        return 0.5*(self.lf_instance(Q, dx, dt) + self.lw_instance(Q, dx, dt))

    def __str__(self):
        return "Force1D"

class LaxFriedrichs1D:
    """
        Generalized LaxFriedrichs flux-operator, as defined in Toro p. 609
    """
    num_ghost_cells = 1
    def __init__(self, F, num_dimensions = 1):
        self.F = F
        self.num_dimensions = num_dimensions
    def __call__(self, Q, dx, dt):
        Q_center = Q[:-1]
        Q_right = Q[1:]

        if USE_NUMEXPR:
            f1 = self.F(Q_center, dx)
            f2 = self.F(Q_right, dx)
            n_dim = self.num_dimensions
            return ne.evaluate("0.5*(f1+f2) - 0.5*(dx/(n_dim*dt))*(Q_right-Q_center)")
        else:
            f1 = self.F(Q_center, dx)
            f2 = self.F(Q_right, dx)
            f3 = Q_right
            f4 = Q_center
            return 0.5*(self.F(Q_center, dx)+self.F(Q_right, dx)) - 0.5*(dx/(self.num_dimensions*dt))*(Q_right-Q_center)

    def __str__(self):
        return "LaxFriedrichs1D"

class GodunovType1D:
    num_ghost_cells = 1
    def __init__(self, F):
        self.F = F
    def __call__(self, Q, dx, dt):
        Q_center = Q[:-1]
        Q_right = Q[1:]
        Q_interface = 0.5*(Q_right + Q_center)
        return self.F(Q_interface, dx)

class Godunov1D:
    num_ghost_cells = 1
    def __init__(self, model):
        self.model = model

    def __call__(self, Q, dx, dt):
        Q_center = Q[:-1]
        Q_right = Q[1:]
        rp_s = self.model.RiemannSolver_Vectorized(q_l = Q_center, q_r = Q_right)

        N = len(Q)-1
        x = np.zeros((N,))

        return self.model.F(rp_s(x), dx)



class LaxWendroff1D:
    num_ghost_cells = 1
    def __init__(self, F):
        self.F = F
    def __call__(self, Q, dx, dt):
        Q_center = Q[:-1]
        Q_right = Q[1:]

        if USE_NUMEXPR:
            f1 = self.F(Q_center,dx)
            f2 = self.F(Q_right,dx)
            Q_half_t_right = ne.evaluate("0.5*(Q_center+Q_right) + 0.5*(dt/dx)*(f1 - f2)")

            return self.F(Q_half_t_right, dx)
        else:
            return 0.5*(Q_center+Q_right) + 0.5*(dt/dx)*(self.F(Q_center,dx) - self.F(Q_right,dx))


    def __str__(self):
        return "LaxWendroff1D (Ritchmyer version)"

class CTU_2D:
    """
    First-order implementation of Corner Transport Upwind method in 2D
    """
    num_ghost_cells = 1

    def __init__(self, model):
        self.model = model
        if not hasattr(model, "RiemannSolver"):
            raise Exception("The current model (%s) does not have a Riemann Solver defined" % model)
        if not hasattr(model,"F") or not hasattr(model, "G"):
            raise Exception("The current model (%s) does not have two flux operations defined" % model)
        if not hasattr(model, "RiemannSolver_Vectorized"):
            import warnings
            warnings.warn("The current model (%s) does not have a vectorized Riemann Solver defined, using fall-back vectorization wrapper" % model)
            self.RiemannSolver = makeWrappedVectorizedRiemannSolver(model.RiemannSolver)
        else:
            self.RiemannSolver = model.RiemannSolver_Vectorized

    def __call__(self, Q, dx, dy, dt):
        model = self.model
        # x-direction
        # 1. Compute transverse flux side and time integrals by solving Riemann problems at the cell sides

        from common import print_grid

        # transverse (y-) direction
        (q_b, q_t) = (Q[:,:-1], Q[:,1:])
        rp_y = self.RiemannSolver(q_l=q_b, q_r=q_t, norm=(0,1))

        # 2. Compute left and right states for each interface, using a Godunov evalution of the states in the above Riemann Problems
        # (denote half-timestep states by _h)
        xts = np.zeros(q_b.shape[:-1]) # rp_y.shape is the number of RPs, each is evaluated at the interface, so xt=0.0
        q_int = rp_y.getState(xt=xts) # evaluate Riemann Problems at interfaces in y-direction
        q_h = Q[:,1:-1] - dt/(2.0*dy)*( model.G(q_int[:,1:], dy) - model.G(q_int[:,:-1], dy) )

        # 3. Solve RP at interface in x-direction
        (q_l, q_r) = (q_h[:-1,:], q_h[1:,:])
        rp_x = self.RiemannSolver(q_l=q_l, q_r=q_r, norm=(1,0))
        xts = np.zeros(q_l.shape[:-1])
        q_int_x = rp_x.getState(xt=xts)

        # Repeat process for y-interfaces
        (q_l, q_r) = (Q[:-1,:], Q[1:,:])
        rp_x = self.RiemannSolver(q_l=q_l, q_r=q_r, norm=(1,0))

        xts = np.zeros(q_l.shape[:-1])
        q_int = rp_x.getState(xt=xts)
        q_h = Q[1:-1,:] - dt/(2.0*dx)*( model.F(q_int[1:,:], dx) - model.F(q_int[:-1,:], dx) )

        (q_b, q_t) = (q_h[:,:-1], q_h[:,1:])
        rp_y = self.RiemannSolver(q_l=q_b, q_r=q_t, norm=(0,1))
        xts = np.zeros(q_b.shape[:-1])
        q_int_y = rp_y.getState(xt=xts)

        # we now have interface states at each interface, at this point we may perform a projection on the velocity field
        
        # and then, time to calculate the fluxes
        f = model.F(q_int_x, dx)
        g = model.G(q_int_y, dy)

        #print Q.shape, q_int_x.shape, q_int_y.shape
        #import sys
        #sys.exit()

        return (f,g)


class SLIC_1D:
    omega = 0.0 
    num_ghost_cells = 2

    def __init__(self, F, model_state):
        self.model_state = model_state
        self.F = F

    def __applyLimiter(q_slope, r, omega):
        return q_slope*self.__slopeLimiter(r, omega)

    def __call__(self, Q, dx, dt):
        # calculate the differences between all cells
        dQ_interface = Q[1:] - Q[:-1]

        # get the variable used for slope calculations, and at the same time make sure that the differences
        # between cells is a least a value epsilon (boolean True evaluates as 1.0, False as 0.0)
        slope_var = dQ_interface.view(self.model_state).slopeCalcVar + (dQ_interface.view(self.model_state).slopeCalcVar == 0.0)*epsilon

        # calculate slope ratios between interface cells, each cell (inside the domain)
        # will now have a slope associated with it
        r = slope_var[:-1]/slope_var[1:]

        # combine the slopes in a weighted fashion, this slope will later be used for interpolation once the slopes have been limited
        q_slope = 0.5*(1.0+self.omega)*dQ_interface[:-1] + 0.5*(1.0-self.omega)*dQ_interface[1:]
        

        # apply the slope limiter (I don't know a neat way of vectorizing this a the moment, though I should a some point, it will be similar
        # to the epsilon operation above). The slopes are limited for each component in turn
        r_limited = (np.vectorize(self.__slopeLimiter))(r, self.omega)
        for i in range(self.model_state.num_components):
            q_slope[...,i] = q_slope[...,i]*r_limited
        
        # extrapolate interface values, these are also needed in the ghost cells, as otherwise we can do the half timestep update later on
        q_l = Q[1:-1] - 0.5*q_slope
        q_r = Q[1:-1] + 0.5*q_slope



        # evolve the interface states by half a timestep
        dq_bar = 0.5*dt/dx*(self.F(q_l,dx) - self.F(q_r,dx))
        q_l_bar = q_l + dq_bar
        q_r_bar = q_r + dq_bar
        

        # Now to the flux calculations, the bar-states must be subscripted as only the states at the interfaces
        # are needed, and so not the leftmost left bar state, and similarly for the right-most right bar state.
        q_l_bar = q_l_bar[1:]
        q_r_bar = q_r_bar[:-1]


        # NB! Left and right states may seemed to be swapped around below, but that is because the flux is being calculated
        # across the interface using the bar-states. Each bar-state calculated through extrapolation to the cell interface
        # This means that the state left of an interface is in fact the right-bar state, and vice versa
        f_lf = 0.5*( self.F(q_r_bar, dx)  +  self.F(q_l_bar, dx) ) + 0.5*(dx/dt)*(q_r_bar-q_l_bar)
        f_rm = self.F(    0.5*(q_r_bar+q_l_bar)   +   0.5*dt/dx*( self.F(q_r_bar,dx) - self.F(q_l_bar,dx) )    ,dx)

        f = 0.5*(f_lf + f_rm)
        
        return f

    def __slopeLimiter(self,r,omega):
         def limiterVanLeer(r, omega):
             e = e_r(r, omega)
             if r <= 0.0:
                 return 0.0
             elif e < 2.0*r/(1.0+r):
                 return e
             else:
                 return 2.0*r / (1.0+r)
         def e_r(r, omega):
             beta = 1.0
             x = (1.0 - omega + (1 + omega)*r)
             if x == 0:
                 return 0.0
             else:
                 return 2.0 * beta / x
         return limiterVanLeer(r, omega)
    
    def __str__(self):
         return "SLIC_1D"


def makeWrappedVectorizedRiemannSolver(baseRiemannSolver):
    class WrappedSolver:
        def __init__(self, q_l, q_r, norm):
            self.input_shape = q_l.shape

            if len(self.input_shape) == 3:
                # 2D problem
                from common import meshgrid
                (i, j) = meshgrid(np.arange(self.input_shape[0]), np.arange(self.input_shape[1]))
                self.rps = np.vectorize( lambda i_,j_:baseRiemannSolver(q_l[i_,j_],q_r[i_,j_],norm) )(i, j)
            else:
                raise Exception("Vectorization wrapper has not be written for the dimensionality of the model used.")

        def getState(self, xt):
            if self.input_shape[:-1] != xt.shape:
                raise Exception("Error wrong number of RP evaluation points passed in.")

            if len(self.input_shape) == 3:
                # 2D problem
                from common import meshgrid
                #(i, j) = meshgrid(np.arange(input_shape[0]), np.arange(input_shape[1]))
                #return np.vectorize( lambda i_,j_:self.rps[i_,j_].getState(xt[i_,j_]) )(i, j)
                solutions = np.empty(self.input_shape)
                for i in range(self.input_shape[0]):
                    for j in range(self.input_shape[1]):
                        solutions[i,j] = self.rps[i,j].getState(xt[i,j])
                return solutions
            else:
                raise Exception("Vectorization wrapper has not be written for the dimensionality of the model used.")
    return WrappedSolver


