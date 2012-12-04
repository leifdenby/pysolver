from grids import boundary_conditions as BCs

class TopHat():
    finalTime = 1.0
    
    def __init__(self, model):
        self.model = model
        self.internal_state = self.model.state.make(phi=1.0)
        self.ambient_state = self.model.state.make(phi=0.0)
        self.halfwidth = 0.1
        
        self.boundary_conditions ={ 0: (BCs.Periodic(), BCs.Periodic(), BCs.Periodic(), BCs.Periodic()),
                            }

    def initialCondition(self, pos):
        [x, y] = pos
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

        pos = [x - self.model.x_vel*t, y - self.model.y_vel*t]

        return self.initialCondition(pos=pos)

    def __str__(self):
        return "Linear advection (top hat)"

