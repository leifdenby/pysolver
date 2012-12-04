import grids.boundary_conditions as BCs

class SmoothThermalBuoyantBubble2D():
    finalTime = 10000.0
    exactSolution = None
    def __init__(self, grid, model, radius = 0.1, T0 = 298.0, dT = 30.0):
        self.model = model
        self.radius = radius
        self.dT = dT
        self.T0 = T0
        self.smoothness = 0.1/grid.getGridSpacing()[0]
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic()) }
        self.grid = grid

    def initialCondition(self, x, y):
        temp = self._calc_temp_at(x, y)
        return self.model.state.make(temp=temp, x_velocity=0.0, y_velocity=0.0)

    def _calc_temp_at(self, x, y):
        from math import sqrt, tanh
        (x0, y0) = self.grid.getCenter()
        r = sqrt((x0-x)**2.0 + (y0-y)**2.0)
        return self.T0 + 0.5*self.dT - (self.dT)/2.0*tanh(self.smoothness*(r - self.radius))

    def __str__(self):
        return "Thermally buoyant bubble (dT=%f, r=%f)" % (self.dT, self.radius)

class SmoothThermalBuoyantBubble2D_Klein():
    finalTime = 10000.0
    exactSolution = None
    def __init__(self, grid, model, radius = 0.1, dT = 30.0):
        self.model = model
        self.radius = radius
        self.dT = dT
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic()) }
        self.grid = grid

    def initialCondition(self, x, y):
        dT = self._calc_temp_at(x, y)
        return self.model.state.make(dT=dT, x_velocity=0.0, y_velocity=0.0)

    def _calc_temp_at(self, x, y):
        from math import sqrt, cos
        from scipy.constants import pi
        (x1, y1) = self.grid.getLengthScales()
        (x0, y0) = self.grid.getCenter()
        L = 10000.0
        r = 5.0*sqrt((x/L)**2.0 + (y/L - 1.0/5.0)**2.0)
        if r > 1.0:
            return 0.0
        else:
            return self.dT * cos(pi/2.0*r)**2.0

    def __str__(self):
        return "Thermally buoyant bubble (dT=%f, r=%f)" % (self.dT, self.radius)

class SmoothThermalBuoyantBubble2D_2():
    """
    Smooth buoyant bubble based on Almgren et al. 2006

    (I stopped using this profile because I don't understand how the delta paramter is related to
    the radius).
    """
    finalTime = 10000.0
    exactSolution = None
    def __init__(self, grid, model, T0 = 298.0, dT = 30.0, smoothness = 20.0):
        self.model = model
        self.T0 = T0
        self.dT = dT
        self.smoothness = smoothness
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic(), BCs.Adiabatic()) }
        self.grid = grid

    def initialCondition(self, x, y):
        temp_pot = self._calc_temp_at(x, y)
        return self.model.state.make(temp_pot=temp_pot, x_velocity=0.0, y_velocity=0.0)

    def _calc_temp_at(self, x, y):
        from math import sqrt, tanh
        (x0, y0) = self.grid.getCenter()
        ep = sqrt((x-x0)**2.0 + (y-y0)**2.0)
        return self.T0 + self.dT/2.0*(1.0 + tanh( (2.0 - ep*self.smoothness)/0.9))

    def __str__(self):
        return "Thermally buoyant bubble (T0=%f, dT=%f)" % (self.T0, self.dT)


class StratifiedAtmosphere():
    def __init__(self, grid, model, T0, dTheta_dz, dT):
        self.model = model
        self.dTheta_dz = dTheta_dz 
        self.boundary_conditions ={ 0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.NoSlip()),
                            1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()), 
                            2: (BCs.Adiabatic(), BCs.Adiabatic(), BCs.Neumann(slope=self.dTheta_dz), BCs.Neumann(slope=self.dTheta_dz)) }
        self.grid = grid
        self.T0 = T0
        self.dT = dT
        self.smoothness = 0.5/grid.getGridSpacing()[0]
        self.radius = 0.05*grid.getLengthScales()[0]
        self.temp_range = (T0, T0+dT)

    def initialCondition(self, x, y):
        from math import sqrt, tanh
        (x0, y0) = self.grid.getCenter()
        r = sqrt((x0-x)**2.0 + (y0-y)**2.0)
        
        temp = self.T0 + y*self.dTheta_dz + 0.5*self.dT - (self.dT)/2.0*tanh(self.smoothness*(r - self.radius))
        #temp = y*self.dTheta_dz + 0.5*self.dT - (self.dT)/2.0*tanh(self.smoothness*(r - self.radius))
        return self.model.state.make(temp=temp, x_velocity=0.0, y_velocity=0.0)

    def __str__(self):
        return "Stable atmosphere dTheta/dz=%eK/m, with bubble perturbation dT=%eK" % (self.dTheta_dz, self.dT)

