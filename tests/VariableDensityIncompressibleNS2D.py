from grids import boundary_conditions as BCs
import numpy as np
import geometry.areaweighting

from math import tanh, sqrt

PROJECT_ROOT = '/Users/leifdenby/Desktop/PhD/low_Mach_code/'

class LidDrivenCavity():
    def __init__(self, domain_spec, u_top, viscosity, rho0=1.0):
        self.u_top = u_top
        self.rho0 = rho0
        self.boundary_conditions = { 
                0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.MovingWall(wall_velocity=u_top)),
                1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()),
                2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient()) 
                }
        self.ic_func = [lambda (x, y): 0.0*x*y, lambda (x, y): 0.0*x*y, lambda (x, y): 0.0*x*y + rho0]

        # FIXME: Take domain_spec aspect-ratio into account with discrete solutions
        self.domain_spec = domain_spec
        self.Re = rho0*u_top*max(domain_spec.getLengthScales())/viscosity
        
        self.__loadDiscreteSolutionData()
    
    def __loadDiscreteSolutionData(self):
        Re_s = [100, 400, 1000, 320, 5000, 7500, 10000]

        if int(self.Re) in Re_s:
            dtype = [('domain_spec_point','i'), ('y','f')]
            for Re in Re_s:
                dtype.append(('Re_%i' % Re, 'f'))

            raw_data_u = np.loadtxt(PROJECT_ROOT + 'lib/tests/data/incompressibleNS/ghia_data/u_vel.dat',dtype=dtype)
            
            dtype = [('domain_spec_point','i'), ('x','f')]
            for Re in Re_s:
                dtype.append(('Re_%i' % Re, 'f'))
            raw_data_v = np.loadtxt(PROJECT_ROOT + 'lib/tests/data/incompressibleNS/ghia_data/v_vel.dat',dtype=dtype)

            self.discrete_solution = {
                    'u_vel' : { 'y' : raw_data_u['y'], 'u' : raw_data_u['Re_%i' % self.Re]},
                    'v_vel' : { 'x' : raw_data_v['x'], 'v' : raw_data_v['Re_%i' % self.Re]},
                    'source' : 'Ghia et al. 1982',
                    }


    def __str__(self):
        return "Lid-driven cavity problem for variable density NS (u_top=%f, rho0=%f, Re=%f)" % (self.u_top, self.rho0, self.Re)

class LidDrivenCavityWithPerturbation():
    def __init__(self, domain_spec, u_top, viscosity, rho0=1.0, rho1=2.0):
        self.u_top = u_top
        self.rho0 = rho0
        self.rho1 = rho1
        self.boundary_conditions = { 
                0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.MovingWall(wall_velocity=u_top)),
                1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()),
                2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient()) 
                }
        self.ic_func = [lambda (x, y): 0.0*x*y, lambda (x, y): 0.0*x*y, lambda (x, y): 0.0*x*y + self.__calcDensity(x,y)]

        self.domain_spec = domain_spec

    def __calcDensity(self, x, y):
        (x0, y0) = self.domain_spec.getCenter()
        (x1, y1) = self.domain_spec.getLengthScales()
        if abs(x-x0) < 0.1*x1 and abs(y-y0) < 0.1*y1:
            return self.rho1
        else:
            return self.rho0

    def __str__(self):
        return "Lid-driven cavity problem for variable density NS with perturbation (u_top=%f, rho0=%f, rho1=%f)" % (self.u_top, self.rho0, self.rho1)

class BuoyantBubble():
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, model, radius, density=1.2):
        self.model = model
        self.internal_state = self.model.state.make(rho=density,x_velocity=0.0, y_velocity=0.0)
        self.ambient_state = self.model.state.make(rho=1.0,x_velocity=0.0, y_velocity=0.0)
        self.radius = radius
        self.boundary_conditions ={ 
                0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())
                }

    def initialCondition(self, x, y):
        if ((x-0.5)**2 + (y-0.5)**2.0) < self.radius**2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):

        return "Buoyant bubble"

class Uniform():
    def __init__(self, model, rho = 1.0):
        self.boundary_conditions = { 
                0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()),
                2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient()) 
                }
        self.model = model
        self.rho = rho

    def initialCondition(self, x, y):
        return self.model.state.make(x_velocity=0.0, y_velocity=0.0, rho=self.rho)

    def __str__(self):
        return "Uniform IC (%f)" % (self.rho)

class LinearAdvection():
    def __init__(self, model, rho1 = 1.0, rho2 = 2.0):
        self.boundary_conditions = { 
                0: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                1: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                2: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                }
        self.model = model
        self.y_vel = 0.0
        self.x_vel = 1.0
        self.rho1 = rho1
        self.rho2 = rho2
        if model.g != 0.0:
            raise Warning("This test is meant for use in g=0.0 conditions")

    def initialCondition(self, x, y):
        if np.abs(x - 0.5) < 0.2 and np.abs(y - 0.5) < 0.2:
            return self.model.state.make(x_velocity=self.x_vel, y_velocity=self.y_vel, rho=self.rho2)
        else:
            return self.model.state.make(x_velocity=self.x_vel, y_velocity=self.y_vel, rho=self.rho1)

    def __str__(self):
        return "Linear advection (%f, %f)" % (self.rho1, self.rho2)

class CenteredRarefaction():
    def __init__(self, model):
        self.boundary_conditions = { 
                0: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                1: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                2: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                }
        self.model = model

    def initialCondition(self, x, y):
        from math import exp
        x_vel = exp(- (x-0.5)**2.0*10.0 )*np.sign(x-0.5)
        y_vel = exp(- (y-0.5)**2.0*10.0 )
        return self.model.state.make(x_velocity=x_vel, y_velocity=y_vel, rho=1.0)

class TwoBubbles():
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, domain_spec, model, radius = 0.1, rho1 = 999.2, rho2 = 1.225, smoothness = 50.0):
        self.model = model
        self.radius = radius
        self.rho1 = rho1
        self.rho2 = rho2
        self.smoothness = smoothness
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())
                            }
        (self.x0, self.y0) = domain_spec.getCenter()
        print domain_spec.getCenter()

    def initialCondition(self, x, y):
        rho = self._calc_density_at(x, y)
        return self.model.state.make(rho=rho, x_velocity=0.0, y_velocity=0.0)

    def _calc_density_at(self, x, y):
        return (self.rho1+self.rho2)/2.0 + (self.rho1-self.rho2)/2.0*tanh(self.smoothness*(sqrt((self.x0*0.5-x)**2.0 + (self.y0-y)**2.0) - self.radius)) + (self.rho1-self.rho2)/2.0*tanh(self.smoothness*(sqrt((self.x0*1.5-x)**2.0 + (self.y0-y)**2.0) - self.radius))



    def __str__(self):
        return "Buoyant bubble (rho1=%f, rho2=%f, r=%f)" % (self.rho1, self.rho2, self.radius)

class SmoothBuoyantBubble():
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, domain_spec, model, ambient_state, d_rho, radius = None, center = None):
        self.model = model
        self.d_rho = d_rho
        if radius is None:
            self.radius = 0.1*domain_spec.getLengthScales()[0]
        else:
            self.radius = radius
        if center is None:
            self.center = domain_spec.getCenter()
        else:
            self.center = center
        self.ambient_state = ambient_state
        self.smoothness = 1.0/domain_spec.getGridSpacing()[0]
        
        bottom_pos = [0, domain_spec.y0]
        top_pos = [0, domain_spec.y1]
        self.boundary_conditions ={ 
                            0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.ZeroGradient(), BCs.ZeroGradient(), 
                                BCs.FixedGradient(gradient=ambient_state.drho_dz(pos=bottom_pos)), 
                                BCs.FixedGradient(gradient=ambient_state.drho_dz(pos=top_pos)), 
                                )
                            }
        
        self.ic_func = [
                        lambda pos: self.ambient_state.vel(pos)[0], 
                        lambda pos: self.ambient_state.vel(pos)[1], 
                        lambda pos: self.ambient_state.rho(pos) + self.__calcDensityPerturbation(pos)
                        ]

    def __calcDensityPerturbation(self, pos):
        (x, y) = (pos[0], pos[1])
        (x0, y0) = self.center
        return (self.d_rho)/2.0 + self.d_rho/2.0*tanh(-self.smoothness*(sqrt((x0-x)**2.0 + (y0-y)**2.0) - self.radius))

    def __str__(self):
        return "Buoyant bubble (d_rho=%f, r=%f) perturbation on\n%s" % (self.d_rho, self.radius, str(self.ambient_state))

class AreaWeightedBubble():
    def __init__(self, domain_spec, model, ambient_state, d_rho, radius = None, center = None):
        self.domain_spec = domain_spec
        self.model = model
        self.d_rho = d_rho
        if radius is None:
            self.radius = 0.1*domain_spec.getLengthScales()[0]
        else:
            self.radius = radius
        if center is None:
            self.center = domain_spec.getCenter()
        else:
            self.center = center
        self.ambient_state = ambient_state
        
        self.phi = geometry.areaweighting.circle_function(center_pos=self.center, radius=self.radius)
        
        bottom_pos = [0, domain_spec.y0]
        top_pos = [0, domain_spec.y1]
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.ZeroGradient(), BCs.ZeroGradient(), 
                                BCs.FixedGradient(gradient=ambient_state.drho_dz(pos=bottom_pos)), 
                                BCs.FixedGradient(gradient=ambient_state.drho_dz(pos=top_pos)), 
                                )
                            }
        
        self.ic_func = [
                        lambda pos: self.ambient_state.vel(pos)[0], 
                        lambda pos: self.ambient_state.vel(pos)[1], 
                        lambda pos: self.ambient_state.rho(pos) + self.__calcDensityPerturbation(pos)
                        ]

    def __calcDensityPerturbation(self, pos):
        (dx, dy) = self.domain_spec.getdomain_specSpacing()
        [x, y] = pos
        vertices = np.array([[x-0.5*dx, y-0.5*dy], [x-0.5*dx,y+0.5*dx], [x+0.5*dx,y+0.5*dy], [x+0.5*dx, y-0.5*dy]])
        fraction_inside = geometry.areaweighting.calc_volumefraction(phi=self.phi, vertices=vertices)

        if fraction_inside > 1.0 or fraction_inside < 0.0:
            print fraction_inside
            print self.center, self.radius, pos
            print vertices
            raise Exception()

        rho = self.ambient_state.rho(pos) + fraction_inside*self.d_rho
        (x_vel, y_vel) = self.ambient_state.vel(pos)
        return rho

    def __str__(self):
        return "Areaweighted Buoyant bubble (d_rho=%f, r=%f) perturbation on\n%s" % (self.d_rho, self.radius, str(self.ambient_state))
