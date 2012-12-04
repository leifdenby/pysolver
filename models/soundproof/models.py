"""
This module contains models for general low Mach number flow, these
models are not defined in terms as perturbations about an ambient state.
"""

import numpy as np
import scipy.constants
from common import aslice

class ConstantDensityIncompressibleNS2D(object):
    """
        Base model class for low Mach number models in 2D
    """
    def __init__(self, viscosity):
        self.viscosity = viscosity
        class Q_vec(np.ndarray):
            num_components = 2
            @staticmethod
            def make(x_velocity, y_velocity):
                vec = np.empty((Q_vec.num_components,)).view(Q_vec)
                vec.x_velocity = x_velocity
                vec.y_velocity = y_velocity
                return vec

            @property
            def x_velocity(self):
                return self[...,0]
            @x_velocity.setter
            def x_velocity(self, value):
                self[...,0] = value

            @property
            def y_velocity(self):
                return self[...,1]
            @y_velocity.setter
            def y_velocity(self, value):
                self[...,1] = value

        self.state = Q_vec

    def getDiffusionCoeffs(self):
        return np.array([self.viscosity, self.viscosity])

    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))

        in the incompressible costant density NS a=c=1, so we
        just return an array of ones of the right shape,
        """

        return 1.0*self.getPressureCoefficients(Q, grid, boundary_conditions, axis)
    
    def getPressureCoefficients(self, Q, grid, boundary_conditions, axis=-1):
        """
        Calculates the pressure coefficient. If axis != -1 then the presure 
        coefficient will be calculate at interfaces in the direction of axis.
        """
        (Nx, Ny) = grid.getNumCells()
        Nx = Nx+1 if axis==0 else Nx
        Ny = Ny+1 if axis==1 else Ny

        return np.ones((Nx,Ny))
        

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        """
        The incompressible NS do not have a buoyancy term, just return zeros so that we may
        group this model with the other low Mach models
        """
        (Nx, Ny) = grid.getNumCells()
        boundary_states = [np.zeros((1, Ny)), np.zeros((1, Ny)), np.zeros((Nx,1)), np.zeros((Nx,1))]
        return boundary_states

    def calculateBuoyancyTerm(self, Q, pos):
        return np.zeros(Q.shape)

class VariableDensityIncompressibleNS2D(ConstantDensityIncompressibleNS2D):
    """
    model class for variable density incompressible Navier-Stokes equations. In
    addition to the x- and y-velocity components the state vector also includes
    the local density. The momentum equation includes a gravitational buoyancy term.
    The divergence of the velocity field is still required to be zero.

    This model is valueable for buoyant flow through density variations, but the
    density variations must be present in the initial condition an will not develop
    from temperature changes, since the model has no temperature dependence.

    The momentum equation is written without approximation, and not with respect
    to a background stratified state.

    In the framework of the low Mach number models the coefficients are (in 
    Nance-Durran notation)

    a = 1/rho(x,y,t), b=g, P=p, c=1, k=0
    """
    def __init__(self, viscosity, diffusion_coefficient, g = None):
        if g == None:
            self.g = scipy.constants.g
        else:
            self.g = g
        super(VariableDensityIncompressibleNS2D, self).__init__(viscosity=viscosity)
        self.diffusion_coefficient = diffusion_coefficient

        class Q_vec(self.state):
            num_components = self.state.num_components+1
            @staticmethod
            def make(x_velocity, y_velocity, rho):
                vec = np.empty((Q_vec.num_components,)).view(Q_vec)
                vec.x_velocity = x_velocity
                vec.y_velocity = y_velocity
                vec.rho = rho
                return vec
            @property
            def rho(self):
                return self[...,2]
            @rho.setter
            def rho(self, value):
                self[...,2] = value

        self.state = Q_vec

    def __str__(self):
        return "VariableDensityIncompressibleNS (g=%f)" % (self.g)
    
    def __repr__(self):
        return "VariableDensityIncompressibleNS"

    def getDiffusionCoeffs(self):
        return np.append(super(VariableDensityIncompressibleNS2D, self).getDiffusionCoeffs(), [self.diffusion_coefficient])

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        """
        The buoyancy term in the variable density incompressible NS is simply b=g and c=1, so
        we return arrays of g.
        """
        (Nx, Ny) = grid.getNumCells()
        boundary_states = [
                                np.zeros((Ny,)), 
                                np.zeros((Ny,)), 
                                self.g*np.ones((Nx)), 
                                -self.g*np.ones((Nx))
                          ]
        return boundary_states
    
    def calculateBuoyancyTerm(self, Q, pos):
        b = np.zeros(Q.shape)
        b[...,1] = -self.g
        return b
    
    def getDivergenceCoefficients(self, pos):
        """
        calculate c in div(c U*)
        """
        
        return 1.0
    
    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))

        in the incompressible variable density NS a=1, c=1/rho, so we need
        rho at the cell interfaces.
        """

        # c*a
        return 1.0*self.getPressureCoefficients(Q=Q, grid=grid, boundary_conditions=boundary_conditions, axis=axis)

    def getPressureCoefficients(self, Q, grid, boundary_conditions, axis=-1):
        """
        Calculates the pressure coefficient. If axis != -1 then the presure 
        coefficient will be calculate at interfaces in the direction of axis.
        """
        n = grid.num_ghost_cells
        if axis == -1:
            return 1.0/Q.view(self.state).rho[n:-n,n:-n]
        else:
            # first we average the states in the right direction (this will include ghost cells)
            Q_avg = grid.makeInterfaceStates(Q, axis, boundary_conditions=boundary_conditions)

            # pick out the states we're actuall interested in
            p_axis = 1 if axis==0 else 0
            from common import aslice
            r_lim = None if n==1 else -(n-1)
            return 1.0/Q_avg.view(self.state).rho[aslice(p_axis,n,-n)][aslice(axis, n-1,r_lim)]
