import models.euler
from common import BoundaryCondition

class ToroTest1():
    finalTime = 0.20
    boundaryCondition = BoundaryCondition.TRANSMISSIVE

    def __init__(self, gamma):
        self.interface_position = 0.3
        self.model = models.euler.Euler1D(gamma)
        self.state_l = self.model.state.make(density=1.0, velocity=0.0, pressure=1.0)
        self.state_r = self.model.state.make(density=0.125, velocity=0.0, pressure=0.1)
        self.riemannSolver = self.model.RiemannSolver(self.state_l, self.state_r)

    def initialCondition(self, x):
        if x < self.interface_position:
            return self.state_l
        else:
            return self.state_r

    def exactSolution(self, x, t):
        if t == 0.0:
            return self.initialCondition(x)
        else:
            return self.riemannSolver.getState((x-self.interface_position)/t)

    def __str__(self):
        return "Sod's test (Toro 1)"


class ToroTest1_2D():
    boundaryCondition = (ToroTest1.boundaryCondition, ToroTest1.boundaryCondition)
    finalTime = ToroTest1.finalTime
    
    def __init__(self, model):
        self.interface_position = 0.3
        self.model = model
        self.state_l = self.model.state.make(density=1.0, x_velocity=0.0, y_velocity=0.0, pressure=1.0)
        self.state_r = self.model.state.make(density=0.125, x_velocity=0.0, y_velocity=0.0, pressure=0.1)

        self.problem_1d = ToroTest1(model.gamma)

    def initialCondition(self, x, y):
        if x < self.interface_position:
            return self.state_l
        else:
            return self.state_r

    def exactSolution(self, x, y, t):
        if t == 0.0:
            return self.initialCondition(x, y)
        else:
            vec_1d = self.problem_1d.riemannSolver.getState((x-self.interface_position)/t).view(self.problem_1d.model.state)
            vec_2d = self.model.state.make(density=vec_1d.density,x_velocity=vec_1d.velocity,y_velocity=0.0,pressure=vec_1d.pressure)
            return vec_2d

    def __str__(self):
        return "Sod's test (Toro 1)"


class LinearAdvection2D():
    boundaryCondition = (BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC)
    finalTime = 1.0
    
    def __init__(self, model, x_velocity = 1.0, y_velocity = 1.0):
        self.model = model
        (self.x_velocity, self.y_velocity) = (y_velocity, y_velocity)
        self.internal_state = self.model.state.make(density=2.450,x_velocity=x_velocity, y_velocity=y_velocity, pressure=1.0)
        self.ambient_state = self.model.state.make(density=1.225,x_velocity=x_velocity, y_velocity=y_velocity, pressure=1.0)
        self.halfwidth = 0.1

    def initialCondition(self, x, y):
        if abs(x-0.5) < self.halfwidth and abs(y-0.5) < self.halfwidth:
            return self.internal_state
        else:
            return self.ambient_state

    def exactSolution(self, x, y, t):
        while x > 1.0:
            x -= 1.0
        while x < 0.0:
            x += 1.0
        while y > 1.0:
            y -= 1.0
        while y < 0.0:
            y += 1.0


        return self.initialCondition(x - self.x_velocity*t, y - self.y_velocity*t)

    def __str__(self):
        return "Linear advection (top hat, x_vel=%f, y_vel=%f)" % (self.x_velocity, self.y_velocity)


class ShearInterface2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 1.0
    
    def __init__(self, gamma):
        self.model = models.euler.Euler2D(gamma)
        self.top_state = self.model.state.make(density=1.0,x_velocity=1.0, y_velocity=0.0, pressure=1.0)
        self.bottom_state = self.model.state.make(density=1.0,x_velocity=0, y_velocity=0.0, pressure=1.0)
        self.halfwidth = 0.2

    def initialCondition(self, x, y):
        if y < 0.5:
            return self.bottom_state
        else:
            return self.top_state

    def exactSolution(self, x, y, t):
        return self.initialCondition(x, y)

    def __str__(self):
        return "Shear interface"

class BuoyantBubble2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 1.0
    exactSolution = None
    
    def __init__(self, gamma):
        self.model = models.euler.Euler2D(gamma)
        self.internal_state = self.model.state.make(density=0.1,x_velocity=0.0, y_velocity=0.0, pressure=1.0)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0, pressure=1.0)
        self.radius = 0.05

    def initialCondition(self, x, y):
        if ((x-0.5)**2 + (y-0.5)**2.0) < self.radius:
            return self.internal_state
        else:
            return self.ambient_state


    def __str__(self):
        return "Buoyant bubble"

class Uniform2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 10.0
    exactSolution = None
    
    def __init__(self, gamma):
        self.model = models.euler.Euler2D(gamma)
        self.ambient_state = self.model.state.make(density=1.0,x_velocity=0.0, y_velocity=0.0, pressure=1.0)

    def initialCondition(self, x, y):
        return self.ambient_state


    def __str__(self):
        return "Uniform"

class StableAtmosphere2D():
    boundaryCondition = (BoundaryCondition.TRANSMISSIVE, BoundaryCondition.TRANSMISSIVE)
    finalTime = 10.0
    exactSolution = None
    
    def __init__(self, gamma, g):
        self.model = models.euler.Euler2D(gamma)
        self.g = g

    def initialCondition(self, x, y):
        # take constant density with height
        rho = 1.0
        p = rho*self.g*y
        if y < 0.0:
            p = 0.0
        ambient_state = self.model.state.make(density=rho, x_velocity=0.0, y_velocity=0.0, pressure=p)
        return ambient_state


    def __str__(self):
        return "Uniform"
