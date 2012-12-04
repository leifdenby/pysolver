import numpy as np

class LinearAdvection1D:
    def __init__(self, vel):
        self.vel = vel

        class ConservedState1D(np.ndarray):
            num_components = 1

            @staticmethod
            def make(phi):
                vec = np.empty((1,)).view(ConservedState1D)
                vec.phi = phi
                return vec

            # Conserved variables
            def conserved_var_getter(id):
                def get_var(self):
                    return self[...,id]
                return get_var
            def conserved_var_setter(id):
                def set_var(self, value):
                    self[...,id] = value
                return set_var

            phi = property(conserved_var_getter(0), conserved_var_setter(0))
            slopeCalcVar = property(conserved_var_getter(0))

        self.state = ConservedState1D

    def F(self, Q, dx):
        return Q*self.vel

class LinearAdvection2D:
    def __init__(self, vel_x, vel_y):
        self.vel_x = vel_x
        self.vel_y = vel_y
            
        class ConservedState2D(np.ndarray):
            num_components = 1
            # Conserved variables
            def conserved_var_getter(id):
                def get_var(self):
                    return self[...,id]
                return get_var
            def conserved_var_setter(id):
                def set_var(self, value):
                    self[...,id] = value
                return set_var

            phi = property(conserved_var_getter(0), conserved_var_setter(0))
            slopeCalcVar = phi
            
            @staticmethod
            def make(phi):
                vec = np.empty((1,)).view(ConservedState2D)
                vec.phi = phi
                return vec

        class LinearAdvection2DRiemannSolver:
            def __init__(self, q_l, q_r, norm):
                """
                    q_l and q_r are arrays of left and right states (could also be top and bottom)
                    norm is a tuple defining the normal direction of the interface (i.e. either (1,0) or (0,1))
                """
                self.q_l = q_l
                self.q_r = q_r
                self.norm = norm

            def getState(self,xt):
                """
                    Evaluate Riemann Problem solution at x/t = xt
                """

                if self.norm == (1,0):
                    if xt > vel_x:
                        return self.q_r
                    else:
                        return self.q_l
                elif self.norm == (0,1):
                    if xt > vel_y:
                        return self.q_r
                    else:
                        return self.q_l
                else:
                    raise Exception("norm direction pased to Riemann Solver is not valid")

        self.state = ConservedState2D
        self.RiemannSolver = LinearAdvection2DRiemannSolver


    def F(self, Q, dx):
        return Q*self.vel_x
    def G(self, Q, dy):
        return Q*self.vel_y



