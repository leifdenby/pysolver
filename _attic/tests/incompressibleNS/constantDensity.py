from common import BoundaryCondition

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

