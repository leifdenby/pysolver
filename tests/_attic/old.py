
class SmoothThermalBuoyantBubble2D():
    finalTime = 10000.0
    exactSolution = None
    def __init__(self, grid, model, radius = 0.1, T0 = 298.0, dT = 30.0, smoothness = 500.0):
        self.model = model
        self.radius = radius
        self.T0 = T0
        self.dT = dT
        self.smoothness = smoothness
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic()) }
        self.grid = grid

    def initialCondition(self, x, y):
        temp_abs = self._calc_temp_at(x, y)
        return self.model.state.make(temp_abs=temp_abs, x_velocity=0.0, y_velocity=0.0)

    def _calc_temp_at(self, x, y):
        from math import sqrt, tanh
        (x0, y0) = self.grid.getCenter()
        ep = sqrt((x-x0)**2.0 + (y-y0)**2.0)
        delta = 0.08
        return self.T0 + self.dT/2.0*(1.0 + tanh( (2.0 - ep/delta)/0.9))

    def __str__(self):
        return "Thermally buoyant bubble (T0=%f, dT=%f, r=%f)" % (self.T0, self.dT, self.radius)

class HeatedCavity:
    def __init__(self, T0, dT, model):
        self.temp_range = (T0 - dT, T0 + dT)
        self.T0 = T0
        self.dT = dT
        self.model = model

    def initialCondition(self, x, t):
        v = self.model.state.make(x_velocity=0.0, y_velocity=0.0, temp=self.T0)
        return v

class BuoyantBubble2D():
    boundaryCondition = (BoundaryCondition.REFLECTIVE, BoundaryCondition.REFLECTIVE)
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, radius, density=0.1):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.internal_state = self.model.state.make(density=density,x_velocity=0.0, y_velocity=0.0)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0)
        self.radius = radius
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.SolidWall(), BCs.SolidWall(), BCs.SolidWall(), BCs.SolidWall()) }

    def initialCondition(self, x, y):
        if ((x-0.5)**2 + (y-0.5)**2.0) < self.radius**2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):

        return "Buoyant bubble"

class SmoothBuoyantBubble2D():
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, radius = 0.1, rho1 = 999.2, rho2 = 1.225, smoothness = 500.0):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.radius = radius
        self.rho1 = rho1
        self.rho2 = rho2
        self.smoothness = smoothness
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.SolidWall(), BCs.SolidWall(), BCs.SolidWall(), BCs.SolidWall()) }


    def initialCondition(self, x, y):
        rho = self._calc_density_at(x, y)
        return self.model.state.make(density=rho, x_velocity=0.0, y_velocity=0.0)

    def _calc_density_at(self, x, y):
        return (self.rho1+self.rho2)/2.0 + (self.rho1-self.rho2)/2.0*tanh(self.smoothness*(sqrt((0.5-x)**2.0 + (y-0.5)**2.0) - self.radius))



    def __str__(self):
        return "Buoyant bubble (rho1=%f, rho2=%f, r=%f)" % (self.rho1, self.rho2, self.radius)
