import numpy as np
import scipy.constants
from common import aslice

from models import ConstantDensityIncompressibleNS2D

class IncompressibleNS2D(ConstantDensityIncompressibleNS2D):
    """
    model class for variable density incompressible Navier-Stokes equations. In
    addition to the x- and y-velocity components the state vector also includes
    the local density. The momentum equation includes a gravitational buoyancy term.
    The divergence of the velocity field is still required to be zero.

    This model is valueable for buoyant flow through density variations, but the
    density variations must be present in the initial condition an will not develop
    from temperature changes, since the model has no temperature dependence.

    The use of the Boussinesq approximation in the momentum equation is optional, the
    ground state will be used as the reference state.

    In the framework of the low Mach number models the coefficients are (in 
    Nance-Durran notation)

    a = 1/rho(x,y,t), b=g, P=p, c=1, k=0
    """
    def __init__(self, viscosity, diffusion_coefficient, ambient_stratification, use_boussinesq = False):
        if hasattr(ambient_stratification, 'g'):
            self.g = getattr(ambient_stratification, 'g')
        else:
            self.g = scipy.constants.g
        super(IncompressibleNS2D, self).__init__(viscosity=viscosity)
        self.diffusion_coefficient = diffusion_coefficient
        self.ambient_stratification = ambient_stratification
        self.use_boussinesq = use_boussinesq

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
        return "VariableDensityIncompressibleNS, perturbation form, %s Boussinesq approximation (g=%f)" % ("with" if self.use_boussinesq else "without", self.g)
    
    def getDiffusionCoeffs(self):
        return np.append(super(IncompressibleNS2D, self).getDiffusionCoeffs(), [self.diffusion_coefficient])

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

        rho = Q.view(self.state).rho 
        d_rho = rho - self.ambient_stratification.rho(pos)
        if self.use_boussinesq:
            rho0 = self.ambient_stratification.rho0
            b[...,1] = -self.g*d_rho/rho0
        else:
            b[...,1] = -self.g*d_rho/rho
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

        if self.use_boussinesq:
            (Nx, Ny) = grid.getNumCells()
            if axis == 0:
                Nx += 1
            elif axis == 1:
                Ny += 1
            return np.ones((Nx, Ny))/self.ambient_stratification.rho0
        else:
            if axis == -1:
                n = grid.num_ghost_cells
                return 1.0/Q.view(self.state).rho[n:-n,n:-n]
            else:
                # first we average the states in the right direction (this will include ghost cells)
                Q_avg = grid.makeInterfaceStates(Q, axis, boundary_conditions=boundary_conditions)

                # pick out the states we're actuall interested in
                p_axis = 1 if axis==0 else 0
                from common import aslice
                return 1.0/Q_avg.view(self.state).rho[aslice(p_axis,2,-2)][aslice(axis, 1,-1)]


class ThermalLowMachBaseModel2D(ConstantDensityIncompressibleNS2D):
    def __init__(self, viscosity, thermal_diffusion):
        self.thermal_diffusion = thermal_diffusion
        super(ThermalLowMachBaseModel2D, self).__init__(viscosity=viscosity)
        class Q_vec(self.state):
            num_components = self.state.num_components+1
            @staticmethod
            def make(x_velocity, y_velocity, temp):
                vec = np.empty((self.state.num_components,)).view(Q_vec)
                vec.x_velocity = x_velocity
                vec.y_velocity = y_velocity
                vec.temp = temp
                return vec
            @property
            def temp(self):
                return self[...,2]
            @temp.setter
            def temp(self, value):
                self[...,2] = value

        self.state = Q_vec

    def getLaplacianCoefficients(self, Q, axis, grid):
        raise Exception("The base class for thermal low Mach number models should not be used directly.")

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        raise Exception("The base class for thermal low Mach number models should not be used directly.")
    
    def getDiffusionCoeffs(self):
        return np.append(super(ThermalLowMachBaseModel2D, self).getDiffusionCoeffs(), [self.thermal_diffusion])

class Boussinesq2D(ThermalLowMachBaseModel2D):
    """
    For the Boussinesq model the temperature-like variable 'temp' is the potential temperature.
    """
    def __init__(self, viscosity, thermal_diffusion, surface_density, surface_temperature, gas_properties, g = None):
        (self.surface_density, self.thermal_expansion) = (surface_density, gas_properties.thermal_expansion())
        self.surface_temperature = surface_temperature

        if g == None:
            self.g = scipy.constants.g
        else:
            self.g = g

        a = 1.0/surface_density
        b = self.thermal_expansion
        super(Boussinesq2D, self).__init__(viscosity=viscosity, thermal_diffusion=thermal_diffusion)

    def __repr__(self):
        return "Boussinesq2D"
    
    def __str__(self):
        return "Boussinesq 2D"

    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))

        in the in the Boussinesq model a=1 and c=1 (but for good measure we get
        a from getPressureCoefficients)
        """

        # c*a
        return 1.0*self.getPressureCoefficients(Q, None, grid, boundary_conditions, axis)
    
    def getDivergenceCoefficients(self, y):
        return 1.0
    
    def getPressureCoefficients(self, Q, y, grid, boundary_conditions, axis=-1):
        """
        Calculates the pressure coefficient. If axis != -1 then the presure 
        coefficient will be calculate at interfaces in the direction of axis.
        
        For the Boussinesq model a=1.0, but we need to make sure that this has the right
        shape.
        """
        
        (Nx, Ny) = grid.getNumCells()
        Nx = Nx+1 if axis==0 else Nx
        Ny = Ny+1 if axis==1 else Ny
        
        return np.ones((Nx, Ny))

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        """
        The buoyancy term in the variable density incompressible NS is simply b=g and c=1, so
        we return arrays of g.
        """

        raise Exception("Is this called at the moment?")
        (Nx, Ny) = grid.getNumCells()
        boundary_states = [
                                np.zeros((1, Ny)), 
                                np.zeros((1, Ny)), 
                                scipy.constants.g*np.ones((Nx,1)), 
                                scipy.constants.g*np.ones((Nx,1))
                          ]
        return boundary_states
    
    def calculateBuoyancyTerm(self, Q, y):
        b = np.zeros(Q.shape)
        #b[...,1] = self.g*self.thermal_expansion*(Q.view(self.state).temp - self.surface_temperature)
        b[...,1] = self.g*self.thermal_expansion*(Q.view(self.state).temp - y*0.001 - 300.0)
        return b

class OguraPhillips2D(ThermalLowMachBaseModel2D):
    def __init__(self, viscosity, thermal_diffusion, surface_density, surface_temperature, gas_properties, dTheta_dz = 0.0, g = None):
        (self.surface_density, self.thermal_expansion) = (surface_density, gas_properties.thermal_expansion())
        self.surface_temperature = surface_temperature
        self.gas_properties = gas_properties

        self.dTheta_dz = dTheta_dz

        if g == None:
            self.g = scipy.constants.g
        else:
            self.g = g

        #   since the pressure at the ground as used as the reference temperature temp_pot = temp_abs at the ground
        #   theta = T(P0/P)^kappa
        pot_temp_ground = self.surface_temperature
        a = gas_properties.cp()*pot_temp_ground
        b = self.g/pot_temp_ground
        
        super(OguraPhillips2D, self).__init__(viscosity=viscosity, thermal_diffusion=thermal_diffusion)

    def __repr__(self):
        return "OguraPhillips2D"

    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))

        in the in the Boussinesq model a=1 and c=1 (but for good measure we get
        a from getPressureCoefficients)
        """

        # c*a
        if axis == -1:
            (x, y) = grid.getCellCenterPositions()
        else:
            (x, y) = grid.getInterfacePositions(axis=axis)
        
        c = self.calcDensity(y)
        return 1.0*self.getPressureCoefficients(Q, grid, boundary_conditions, y, axis)
    
    def getDivergenceCoefficients(self, y):
        c = self.calcDensity(y)
        return c


    
    def getPressureCoefficients(self, Q, grid, boundary_conditions, y, axis=-1):
        """
        Calculates the pressure coefficient. If axis != -1 then the presure 
        coefficient will be calculate at interfaces in the direction of axis.
        
        For the Boussinesq model a=1.0, but we need to make sure that this has the right
        shape.
        """
        
        (Nx, Ny) = grid.getNumCells()
        Nx = Nx+1 if axis==0 else Nx
        Ny = Ny+1 if axis==1 else Ny
        
        return self.surface_temperature * self.gas_properties.cp() * np.ones((Nx, Ny))

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        """
        The buoyancy term in the variable density incompressible NS is simply b=g and c=1, so
        we return arrays of g.
        """

        raise Exception("Is this called at the moment?")
        # The below is actually wrong, this from the variable density model
        (Nx, Ny) = grid.getNumCells()
        boundary_states = [
                                np.zeros((1, Ny)), 
                                np.zeros((1, Ny)), 
                                scipy.constants.g*np.ones((Nx,1)), 
                                scipy.constants.g*np.ones((Nx,1))
                          ]
        return boundary_states
    
    def calculateBuoyancyTerm(self, Q, y):
        b = np.zeros(Q.shape)
        #b[...,1] = self.g/self.surface_temperature*(Q.view(self.state).temp - y*0.0001 - self.surface_temperature)
        b[...,1] = self.g/self.surface_temperature*(Q.view(self.state).temp - y*self.dTheta_dz - self.surface_temperature)
        #b[...,1] = self.g/self.surface_temperature*Q.view(self.state).temp

        return b

    def calcDensity(self, y):
        rho0 = self.surface_density
        g = scipy.constants.g
        T0 = self.surface_temperature
        cp = self.gas_properties.cp()
        cv_m = self.gas_properties.cv_m()
        R = scipy.constants.R
        return rho0*(1.0 - g*y/(cp*T0))**(cv_m/R)

class WillhelmsonOgura2D(ThermalLowMachBaseModel2D):
    def __init__(self, viscosity, thermal_diffusion, surface_density, surface_temperature, gas_properties, dTheta_dz, g = None):
        (self.surface_density, self.thermal_expansion) = (surface_density, gas_properties.thermal_expansion())
        self.surface_temperature = surface_temperature
        self.gas_properties = gas_properties
        self.dTheta_dz = dTheta_dz

        if g == None:
            self.g = scipy.constants.g
        else:
            self.g = g

        #   since the pressure at the ground as used as the reference temperature temp_pot = temp_abs at the ground
        #   theta = T(P0/P)^kappa
        pot_temp_ground = self.surface_temperature
        a = gas_properties.cp()*pot_temp_ground
        b = self.g/pot_temp_ground
        
        super(WillhelmsonOgura2D, self).__init__(viscosity=viscosity, thermal_diffusion=thermal_diffusion)

    def __repr__(self):
        return "WillhemlsonOgura2D"

    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))

        in the in the Boussinesq model a=1 and c=1 (but for good measure we get
        a from getPressureCoefficients)
        """

        # c*a
        if axis == -1:
            (x, y) = grid.getCellCenterPositions()
        else:
            (x, y) = grid.getInterfacePositions(axis=axis)
        
        c = self.calcDensity(y)
        return 1.0*self.getPressureCoefficients(Q, grid, boundary_conditions, y, axis)
    
    def getPressureCoefficients(self, Q, grid, boundary_conditions, y, axis=-1):
        """
        Calculates the pressure coefficient. If axis != -1 then the presure 
        coefficient will be calculate at interfaces in the direction of axis.
        
        For the Boussinesq model a=1.0, but we need to make sure that this has the right
        shape.
        """
        
        if axis == -1:
            n = 2
            temp = Q.view(self.state).temp[n:-n,n:-n]
        else:
            # The absolute potential temperature has been requsted at the cell interface, so we need to do some interpolation
            p_axis = 1 if axis == 0 else 0
            Q_interface = grid.makeInterfaceStates(Q=Q[aslice(axis,1,-1)][aslice(p_axis,2,-2)], axis=axis)
            temp = Q_interface.view(self.state).temp
        
        return temp * self.gas_properties.cp()

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        """
        The buoyancy term in the variable density incompressible NS is simply b=g and c=1, so
        we return arrays of g.
        """

        raise Exception("Is this called at the moment?")
        # The below is actually wrong, this from the variable density model
        (Nx, Ny) = grid.getNumCells()
        boundary_states = [
                                np.zeros((1, Ny)), 
                                np.zeros((1, Ny)), 
                                scipy.constants.g*np.ones((Nx,1)), 
                                scipy.constants.g*np.ones((Nx,1))
                          ]
        return boundary_states
    
    def calculateBuoyancyTerm(self, Q, y):
        b = np.zeros(Q.shape)
        stratified_temp = self.calcAmbientPotTemp(y)
        b[...,1] = self.g*(Q.view(self.state).temp - stratified_temp)/stratified_temp
        return b

    def calcDensity(self, y):
        rho0 = self.surface_density
        g = scipy.constants.g
        T0 = self.surface_temperature
        cp = self.gas_properties.cp()
        cv_m = self.gas_properties.cv_m()
        R = scipy.constants.R
        return rho0*(1.0 - g*y/(cp*T0))**(cv_m/R)
    
    def calcAmbientPotTemp(self, y):
        return self.surface_temperature + self.dTheta_dz * y

class PseudoIncompressible2D(ThermalLowMachBaseModel2D):
    def __init__(self, viscosity, thermal_diffusion, surface_density, surface_temperature, gas_properties, dTheta_dz, g = None):
        (self.surface_density, self.thermal_expansion) = (surface_density, gas_properties.thermal_expansion())
        self.surface_temperature = surface_temperature
        self.gas_properties = gas_properties
        self.dTheta_dz = dTheta_dz

        if g == None:
            self.g = scipy.constants.g
        else:
            self.g = g

        #   since the pressure at the ground as used as the reference temperature temp_pot = temp_abs at the ground
        #   theta = T(P0/P)^kappa
        pot_temp_ground = self.surface_temperature
        a = gas_properties.cp()*pot_temp_ground
        b = self.g/pot_temp_ground
        
        super(PseudoIncompressible2D, self).__init__(viscosity=viscosity, thermal_diffusion=thermal_diffusion)

    def __repr__(self):
        return "PseudoIncompressible2D"

    def getLaplacianCoefficients(self, Q, axis, grid, boundary_conditions):
        """
        calculate ca in div(ca grad(phi))
        """

        if axis == -1:
            (x, y) = grid.getCellCenterPositions()
        else:
            (x, y) = grid.getInterfacePositions(axis=axis)
        
        c = self.calcAmbientDensity(y)*self.calcAmbientPotTemp(y=y)
        a = self.getPressureCoefficients(Q, grid, boundary_conditions, y, axis)
        return c*a
    
    def getDivergenceCoefficients(self, y):
        """
        calculate c in div(c U*)
        """
        
        c = self.calcAmbientDensity(y)*self.calcAmbientPotTemp(y=y)
        return c

    
    
    def getPressureCoefficients(self, Q, grid, boundary_conditions, y, axis=-1):
        """
        """
        
        (Nx, Ny) = grid.getNumCells()
        Nx = Nx+1 if axis==0 else Nx
        Ny = Ny+1 if axis==1 else Ny
        
        if axis == -1:
            n = 2
            temp = Q.view(self.state).temp[n:-n,n:-n]
        else:
            # The absolute potential temperature has been requsted at the cell interface, so we need to do some interpolation
            p_axis = 1 if axis == 0 else 0
            #print "GOOO ", Q.shape
            dTheta_dz = self.dTheta_dz
            import grids.boundary_conditions as BCs
            boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Neumann(slope=dTheta_dz), BCs.Neumann(slope=dTheta_dz))
                            #2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic())
                            }
            Q_interface = grid.makeInterfaceStates(Q=Q[aslice(axis,1,-1)][aslice(p_axis,2,-2)], axis=axis, boundary_conditions=boundary_conditions)
            #from matplotlib import pyplot as plot
            #plot.figure(12)
            #plot.clf()
            #print Q[Ny/2,:,2].shape, Q_interface[Ny/2,:,2].shape
            #plot.plot(Q_interface[Ny/2,:,2], label='interface')
            #plot.plot(Q[Ny/2,:,2])
            #plot.legend()
            #plot.draw()
            #raw_input()
            temp = Q_interface.view(self.state).temp


        return self.gas_properties.cp() * temp

    def calculateBuoyancyTermOnBoundary(self, Q, grid):
        raise Exception("Is this called at the moment?")
    
    def calculateBuoyancyTerm(self, Q, y):
        b = np.zeros(Q.shape)
        stratified_temp = self.calcAmbientPotTemp(y)
        b[...,1] = self.g*(Q.view(self.state).temp - stratified_temp)/stratified_temp
        return b

    def calcAmbientDensity(self, y):
        rho0 = self.surface_density
        g = scipy.constants.g
        T0 = self.surface_temperature
        cp = self.gas_properties.cp()
        cv_m = self.gas_properties.cv_m()
        R = scipy.constants.R
        return rho0*(1.0 - g*y/(cp*T0))**(cv_m/R)

    def calcAmbientPotTemp(self, y):
        return self.surface_temperature + self.dTheta_dz * y

def test():
    surface_density = 1.225
    surface_temperature = 293.0
    gas_properties = AtmosphericAir()

    bous = Boussinesq2D(surface_density, gas_properties)
    op = OguraPhillips2D(surface_density, surface_temperature, gas_properties)

    from grids import FVGrid2D
    grid = FVGrid2D(N = (10,10), x=(0.0, 1.0), y=(0.0, 10000.0), num_ghost_cells=2)

    rho_b = bous.getLaplacianCoefficients(grid, num_ghost_cells=0)
    rho_op = op.getLaplacianCoefficients(grid, num_ghost_cells=0)

    import matplotlib.pyplot as plot
    plot.ion()

    (x, y) = grid.getInterfacePositions(axis=1, num_ghost_cells=0)
    (x_min, x_max, y_min, y_max) = grid.getExtent()

    plot.plot(rho_b[1][0,:], y[0,:], color="blue", label="Boussinesq")
    plot.plot(rho_op[1][0,:], y[0,:], color="red", label="OguraPhillips")
    plot.xlabel("c (divergence coefficient)")
    plot.ylabel("height/m")
    plot.xlim(0.5, 1.3)
    plot.legend()
    plot.grid(True)
    plot.ylim(y_min, y_max)
    plot.draw()

    raw_input()

class IdealGas(object):
    """
        Class for describing general ideal gas.
    """
    def __init__(self, f, M):
        """
            f: degrees of freedom
            M: molar weight (g/mol)
        """
        self.f = f
        self.M = M

    def cv_m(self):
        """
        molar specific heat at constant volume
        """
        return self.f/2.0*scipy.constants.R

    def cp_m(self):
        """
        molar specific heat at constant pressure
        """
        return self.cv_m() + scipy.constants.R

    def cv(self):
        """
        mass specific heat at constant volume

        units of J/K/kg
        """
        return self.cv_m()/self.M*1000.0

    def cp(self):
        """
        mass specific heat at constant pressure

        units of J/K/kg
        """
        return self.cp_m()/self.M*1000.0

    def thermal_expansion(self):
            #  TODO: insert temperature dependent thermal expansion
        return 3.5e-3 #  1.0/temp

    def gamma(self):
        return self.cp()/self.cv()

    def kappa(self):
        return scipy.constants.R/self.cp()

class DiatomicGas(IdealGas):
    def __init__(self, M):
        super(DiatomicGas, self).__init__(f=5.0, M=M)

class AtmosphericAir(DiatomicGas):
    def __init__(self):
        super(AtmosphericAir, self).__init__(M=28.97)



if __name__ == "__main__":
    test()
