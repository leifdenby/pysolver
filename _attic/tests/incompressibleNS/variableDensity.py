from common import BoundaryCondition
import models.incompressibleNS.variableDensity
from numpy import tanh, sqrt

class Uniform2D():
    boundaryCondition = (BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC)
    finalTime = 1.0
    exactSolution = None
    
    def __init__(self, ambient_state):
        self.ambient_state = ambient_state

    def initialCondition(self, x, y):
        return self.ambient_state

    def __str__(self):
        return "Uniform (%s)" % (self.ambient_state)

class LinearAdvection2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 10.0
    exactSolution = None
    
    def __init__(self, x_velocity = 1.0, y_velocity = 1.0):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.internal_state = self.model.state.make(density=1.225,x_velocity=x_velocity, y_velocity=y_velocity)
        self.ambient_state = self.model.state.make(density=999.0,x_velocity=x_velocity, y_velocity=y_velocity)
        self.width = 0.2

    def initialCondition(self, x, y):
        if abs(x-0.5) < self.width/2.0 and abs(y-0.5) < self.width/2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):
        return "Linear Advection"


class BuoyantBubble2D():
    boundaryCondition = (BoundaryCondition.REFLECTIVE, BoundaryCondition.REFLECTIVE)
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, radius, density=0.1):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.internal_state = self.model.state.make(density=density,x_velocity=0.0, y_velocity=0.0)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0)
        self.radius = radius

    def initialCondition(self, x, y):
        if ((x-0.5)**2 + (y-0.5)**2.0) < self.radius**2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):
        return "Buoyant bubble"

class Interface2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 10.0
    exactSolution = None
    
    def __init__(self):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.lower_state = self.model.state.make(density=0.1,x_velocity=0.0, y_velocity=1.0)
        self.upper_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=1.0)
        self.radius = 0.05

    def initialCondition(self, x, y):
        if y < 0.5:
            return self.lower_state
        else:
            return self.upper_state


    def __str__(self):
        return "Interface"

class BuoyantSquare2D():
    boundaryCondition = (BoundaryCondition.REFLECTIVE, BoundaryCondition.REFLECTIVE)
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.internal_state = self.model.state.make(density=0.1,x_velocity=0.0, y_velocity=0.0)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0)
        self.width = 0.1

    def initialCondition(self, x, y):
        if abs(x-0.5) < self.width/2.0 and abs(y-0.5) < self.width/2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):
        return "Buoyant square"

class BuoyantBand2D():
    boundaryCondition = (BoundaryCondition.REFLECTIVE, BoundaryCondition.REFLECTIVE)
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.internal_state = self.model.state.make(density=0.1,x_velocity=0.0, y_velocity=0.0)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0)
        self.width = 0.3

    def initialCondition(self, x, y):
        if abs(y-0.5) < self.width/2.0:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):
        return "Buoyant band"



class SmoothBuoyantBubble2D():
    finalTime = 10000.0
    exactSolution = None
    
    def __init__(self, radius = 0.1, rho1 = 999.2, rho2 = 1.225, smoothness = 500.0):
        self.model = models.incompressibleNS.variableDensity.VariableDensity2D()
        self.radius = radius
        self.rho1 = rho1
        self.rho2 = rho2
        self.smoothness = smoothness


    def initialCondition(self, x, y):
        rho = self._calc_density_at(x, y)
        return self.model.state.make(density=rho, x_velocity=0.0, y_velocity=0.0)

    def _calc_density_at(self, x, y):
        return (self.rho1+self.rho2)/2.0 + (self.rho1-self.rho2)/2.0*tanh(self.smoothness*(sqrt((0.5-x)**2.0 + (y-0.5)**2.0) - self.radius))



    def __str__(self):
        return "Buoyant bubble (rho1=%f, rho2=%f, r=%f)" % (self.rho1, self.rho2, self.radius)
