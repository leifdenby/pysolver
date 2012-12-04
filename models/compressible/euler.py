import unittest
from numpy import sqrt, fabs, exp, log
from numMethods import rootFinder
import numpy as np
import numexpr as ne

epsilon = 1e-7

class Euler1D:
    """
        Model class for Euler 1D. Contains state vector (U_vec) and flux evaluation method.

        Flux method is evaluated using numexpr for faster processing.

        Commented out is a legacy Riemann Solver, this has not been parallelized for operation
        on numpy-type grids

        To get a grid variable use the Euler1D.state view, e.g. 
            Q = np.random.random((1000,3))
            Q.view(Euler1D_instance).density
    """

    def __init__(self, gamma):
        self.gamma = gamma
    
        class U_vec(np.ndarray):
            """
                View class for use with 2D np.ndarray type grid, (e.g. with shape(Nx,3))
            """
            num_components = 3
            @staticmethod
            def make(density, momentum = None, energy = None, velocity = None, pressure = None):
                vec = np.empty((3,)).view(U_vec)
                if density is not None and momentum is not None and energy is not None:
                    # Internally used variables
                    vec.density = density
                    vec.momentum = momentum
                    vec.energy = energy
                elif density is not None and velocity is not None and pressure is not None:
                    # Calls to these variables are mapped internally
                    # NB! Order of these calls are important, e.g. need density to able to calculate velocity from momentum
                    vec.density = density
                    vec.set_velocity(velocity)
                    vec.set_pressure(pressure)
                elif density is None:
                    vec.density = 0.0
                    vec.momentum = 0.0
                    vec.energy = 0.0
                else:
                    raise TypeError("To define a state you must pass in at least (density,momentum,energy) or (density,velocity,pressure) using kwargs")
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

            density = property(conserved_var_getter(0), conserved_var_setter(0))
            momentum = property(conserved_var_getter(1), conserved_var_setter(1))
            energy = property(conserved_var_getter(2), conserved_var_setter(2))

            slopeCalcVar = property(conserved_var_getter(0))

            # Derived variables
            def get_velocity(self):
                return self.momentum/self.density

            def set_velocity(self, value):
                self.momentum = self.density*value
            
            def get_pressure(self):
                #return (gamma - 1.0)*(self.energy - 0.5*self.momentum**2/self.density)
                E = self.energy
                density = self.density
                momentum = self.momentum
                g = gamma
                #return (gamma - 1.0)*(self.energy - 0.5*self.momentum**2/self.density)
                return ne.evaluate("(g - 1.0)*(E - 0.5*momentum**2/density)")
                
            def set_pressure(self, value):
                self.energy = 0.5*self.momentum**2/self.density + value/(gamma-1.0)

            def get_internal_energy(self):
                return self.pressure/((gamma-1.0)*self.density)

            velocity = property(get_velocity, set_velocity)
            pressure = property(get_pressure, set_pressure)
            internal_energy = property(get_internal_energy)


            def getMaxWaveSpeed(self):
                return np.max(np.abs(self.velocity) + np.sqrt(gamma*self.pressure/self.density))


            def getSlopeCalcVar(self):
                return self.density

        def F(Q, dx):
            vec = Q.view(U_vec)
            mo = vec.momentum
            v = vec.velocity
            p = vec.pressure
            E = vec.energy

            #u1 = vec.momentum
            #u2 = vec.momentum*vec.velocity + vec.pressure
            #u3 = vec.velocity*(vec.energy+vec.pressure)
            
            #u1 = vec.momentum
            #u2 = vec.momentum*vec.velocity + vec.pressure
            #u3 = vec.velocity*(vec.energy+vec.pressure)

            u1 = mo
            u2 = ne.evaluate("mo*v + p")
            u3 = ne.evaluate("v*(E+p)")
            return np.vstack((u1,u2,u3)).T

        class EulerUtils:
            @staticmethod
            def getLeftStateForRightShock(w_right, S_3):
                """
                    Toro p. 101
                """
                (rho_r, u_r, p_r) = (w_right.density, w_right.velocity, w_right.pressure)

                a_r = sqrt(p_r*gamma/rho_r)
                M_r = u_r/a_r

                M_s = S_3/a_r

                rho_s = rho_r*((gamma+1.0)*(M_r-M_s)**2.0)/((gamma-1.0)*(M_r-M_s)**2.0+2.0)
                p_s = p_r*(2.0*gamma*(M_r-M_s)**2.0-(gamma-1.0))/(gamma+1.0)
                u_s = (1.0-rho_r/rho_s)*S_3 + u_r*rho_r/rho_s

                return U_vec(density=rho_s, velocity=u_s, pressure=p_s)

            @staticmethod
            def getRightStateForLeftShock(w_left, S_1):
                """
                    Toro p. 102
                """
                (rho_l, u_l, p_l) = (w_left.density, w_left.velocity, w_left.pressure)

                a_l = sqrt(p_l*gamma/rho_l)
                M_l = u_l/a_l

                M_s = S_1/a_l

                rho_s = rho_l*((gamma+1.0)*(M_l-M_s)**2.0)/((gamma-1.0)*(M_l-M_s)**2.0+2.0)
                p_s = p_l*(2.0*gamma*(M_l-M_s)**2.0-(gamma-1.0))/(gamma+1.0)
                u_s = (1.0-rho_l/rho_s)*S_1 + u_l*rho_l/rho_s
                
                return U_vec(density=rho_s, velocity=u_s, pressure=p_s)
        
        class Euler1DRiemannSolver:
            def __init__(self,state_l,state_r):
                self.gamma = gamma

                self.state_l = state_l
                self.state_r = state_r

                (rho_l, p_l, u_l, rho_r, p_r, u_r) = (state_l.density, state_l.pressure, state_l.velocity, state_r.density, state_r.pressure, state_r.velocity)

                a_l = self.soundSpeed(rho_l, p_l)
                a_r = self.soundSpeed(rho_r, p_r)

                (p_s, u_s) = self.__findPStarAndUStar()
                self.p_s = p_s
                self.u_s = u_s

                gammaRatio = (gamma-1.0)/(gamma+1.0)


                # Speed of contact wave 
                self.S_2 = u_s

                if self.hasLeftShock():
                    rho_s_l = rho_l*( ((p_s/p_l) + (gamma-1.0)/(gamma+1.0)) / ( ((gamma-1.0)/(gamma+1.0))*(p_s/p_l) + 1.0))
                    self.S_L = u_l - a_l*sqrt( (gamma+1.0)/(2.0*gamma)*(p_s/p_l) + (gamma-1.0)/(2.0*gamma) )
                else:
                    rho_s_l = rho_l*(p_s/p_l)**(1.0/gamma)
                    a_s_l = a_l * (p_s/p_l)**((gamma-1.0)/(2.0*gamma))
         
                    # Head of the left rarefaction
                    self.S_HL = u_l - a_l
                    # Tail of the left rarefaction 
                    self.S_TL = u_s - a_s_l
                
                self.state_l_s = U_vec.make(density=rho_s_l,pressure=p_s,velocity=u_s)

                if self.hasRightShock():
                    rho_s_r = rho_r*( ((p_s/p_r) + (gamma-1.0)/(gamma+1.0)) / ( ((gamma-1.0)/(gamma+1.0))*(p_s/p_r) + 1.0))
                    self.S_R = u_r + a_r*sqrt( (gamma+1.0)/(2.0*gamma)*(p_s/p_r) + (gamma-1.0)/(2.0*gamma) )
                else:
                    rho_s_r = rho_r*(p_s/p_r)**(1.0/gamma)
                    a_s_r = a_r * (p_s/p_r)**((gamma-1.0)/(2.0*gamma))
                    # Head of the right rarefaction
                    self.S_HR = u_r + a_r
                    # Tail of the right rarefaction
                    self.S_TR = u_s + a_s_r
                
                self.state_r_s = U_vec.make(density=rho_s_r,pressure=p_s,velocity=u_s)


            def getStates(self, dtdx = None):
                states = [self.state_l]

                if self.hasLeftShock():
                    states.append(self.state_l_s)
                else:
                    if self.leftRarefactionIsSonic():
                        # Add central state, time=0.1 is not important, just have to make sure we don't devide by zero in dx/dt
                        x = 1.0
                        t = x*dtdx
                        states.append(self.getState(x/t))
                    else:
                        states.append(self.state_l_s)

                if self.hasRightShock():
                    states.append(self.state_r_s)
                else:
                    if self.rightRarefactionIsSonic():
                        x = 1.0
                        t = x*dtdx
                        states.append(self.getState(x/t))
                    else:
                        states.append(self.state_r_s)

                states.append(self.state_r)
                
                return states
            
            def getWaveSpeeds(self):
                waveSpeeds = []

                if self.hasLeftShock():
                    waveSpeeds.append(self.S_L)
                else:
                    waveSpeeds.append(self.S_HL)

                waveSpeeds.append(self.S_2)

                if self.hasRightShock():
                    waveSpeeds.append(self.S_R)
                else:
                    waveSpeeds.append(self.S_HR)

                return waveSpeeds



            def soundSpeed(self,density,pressure):
                return sqrt(self.gamma*pressure/density)


            def __findPStarAndUStar(self):
                (rho_r, rho_l, u_r, u_l, p_r, p_l, gamma) = (self.state_r.density, self.state_l.density, self.state_r.velocity, self.state_l.velocity, self.state_r.pressure, self.state_l.pressure, self.gamma)
                
                a_l = self.soundSpeed(rho_l, p_l)
                a_r = self.soundSpeed(rho_r, p_r)
                
                def A_k(rho_k):
                    return 2.0/((self.gamma + 1.0)*rho_k)

                def B_k(p_k):
                    return (gamma - 1.0)/(gamma + 1.0)*p_k

                def f_k(p, state_k):
                    """
                        Expression used to relate density and pressure in front of rarefaction/shock (state_k) to pressure inside (p)
                    """
                    (rho_k, p_k) = (state_k.density, state_k.pressure)
                    if p > p_k:
                        # shock
                        return (p-p_k)*sqrt(A_k(rho_k)/(p + B_k(p_k) ) )
                    else:
                        # rarefaction
                        return 2.0*self.soundSpeed(rho_k,p_k)/(gamma-1.0) * ( (p/p_k)**((gamma-1.0)/(2.0*gamma)) - 1.0 )

                def f(p):
                    """
                        Polynomial to solve, f=0, to find pressure in the star region
                    """
                    return f_k(p, self.state_l) + f_k(p, self.state_r) + self.state_r.velocity - self.state_l.velocity

                def dfdp(p):
                    f_d_value = 0.0
                    if p > p_l:
                        f_d_value += sqrt(A_k(rho_l)/(B_k(p_l)+p))*(1.0-(p-p_l)/(2.0*(B_k(p_l)+p)))
                    else:
                        f_d_value += 1.0/(rho_l*a_l)*(p/p_l)**(-(gamma+1.0)/(2.0*gamma))

                    if p > p_r:
                        f_d_value += sqrt(A_k(rho_r)/(B_k(p_r)+p))*(1.0-(p-p_r)/(2.0*(B_k(p_r)+p)))
                    else:
                        f_d_value += 1.0/(rho_r*a_r)*(p/p_r)**(-(gamma+1.0)/(2.0*gamma))
                    
                    return f_d_value


                def getUStar(p_s):
                    return 0.5*(u_l + u_r) + 0.5*(f_k(p_s, self.state_r) - f_k(p_s, self.state_l) )
                    
                

                k = (gamma-1.0)/(2.0*gamma)

                # double p_tr = pow((a_l + a_r - 0.5*(gamma-1.0)*(u_r-u_l))/(a_l/pow(p_l, (gamma-1.0)/(2.0*gamma)) + a_r/(pow(p_r, (gamma-1.0)/(2.0*gamma)))), 2*gamma/(gamma-1.0));
                p_tr = ( (a_l+a_r-0.5*(gamma-1.0)*(u_r-u_l))/(a_l/(p_l**k)+a_r/(p_r**k)) )**(1.0/k)
                p_s = rootFinder.NewtonRaphson().solve(epsilon, p_tr, f, dfdp)
                    
                return (p_s, getUStar(p_s))


            def hasLeftShock(self):
                return self.state_l.pressure < self.p_s
            
            def hasRightShock(self):
                return self.state_r.pressure < self.p_s

            def hasLeftRarefaction(self):
                return self.state_l.pressure > self.p_s
            
            def hasRightRarefaction(self):
                return self.state_r.pressure > self.p_s

            def leftRarefactionIsSonic(self):
                return self.hasLeftRarefaction() and self.S_HL < 0.0 and self.S_TL > 0.0

            def rightRarefactionIsSonic(self):
                return self.hasRightRarefaction() and self.S_TR < 0.0 and self.S_HR > 0.0

            def hasContactWave(self):
                return abs(self.state_l_s.density - self.state_r_s.density) > 2.0*epsilon
                

            def __str__(self):
                debugLines = []
                debugLines.append("%s, %s, %s, %s" % (self.state_l, self.state_l_s, self.state_r_s, self.state_r))
                if (self.hasLeftShock()):
                    debugLines.append("Left shock: \t\tS_L=%f" % self.S_L)
                elif self.hasLeftRarefaction():
                    debugLines.append("Left rarefaction: \t\tS_HL=%f, S_TL=%f" % (self.S_HL, self.S_TL))
                
                if self.hasContactWave():
                    debugLines.append("Contactwave: \t\tS_2=%f" % self.S_2)

                if self.hasRightShock():
                    debugLines.append("Right shock: \t\tS_R=%f" % self.S_R)
                elif self.hasRightRarefaction():
                    debugLines.append("Right rarefaction: \tS_TR=%f, S_HR=%f" % (self.S_TR, self.S_HR))
                 
                return "\n".join(debugLines)

            def getState(self, xt):
                dxdt = xt

                if dxdt < self.S_2:
                    if self.hasLeftShock():
                        if dxdt < self.S_L:
                            return self.state_l
                        else:
                            return self.state_l_s
                    else:
                        if dxdt < self.S_HL:
                            return self.state_l
                        elif dxdt < self.S_TL:
                            (rho_l, u_l, p_l, gamma) = (self.state_l.density, self.state_l.velocity, self.state_l.pressure, self.gamma)
                            a_l = self.soundSpeed(rho_l, p_l)
                            rho = rho_l*(2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*a_l)*(u_l-dxdt))**(2.0/(gamma-1.0))
                            u = 2.0/(gamma+1.0)*(a_l + (gamma-1.0)/2.0*u_l + dxdt)
                            p = p_l*(2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*a_l)*(u_l - dxdt))**(2.0*gamma/(gamma-1.0))
                            return U_vec.make(density=rho, velocity=u, pressure=p)
                        else:
                            return self.state_l_s
                else:
                    if self.hasRightShock():
                        if dxdt > self.S_R:
                            return self.state_r
                        else:
                            return self.state_r_s
                    else:
                        if dxdt > self.S_HR:
                            return self.state_r
                        elif dxdt > self.S_TR:
                            (rho_r, u_r, p_r, gamma) = (self.state_r.density, self.state_r.velocity, self.state_r.pressure, self.gamma)
                            a_r = self.soundSpeed(rho_r, p_r)
                            rho = rho_r*(2.0/(gamma+1.0)-(gamma-1.0)/((gamma+1.0)*a_r)*(u_r-dxdt))**(2.0/(gamma-1.0))
                            u = 2.0/(gamma+1.0)*(-a_r+(gamma-1.0)/2.0*u_r+dxdt)
                            p = p_r*(2.0/(gamma+1.0)-(gamma-1.0)/((gamma+1.0)*a_r)*(u_r-dxdt))**(2.0*gamma/(gamma-1.0))
                            return U_vec.make(density=rho,velocity=u,pressure=p)
                            
                        else:
                            return self.state_r_s
                
        self.RiemannSolver = Euler1DRiemannSolver
        self.Utils = EulerUtils
        
        self.state = U_vec
        self.F = F


    def __str__(self):
        return "Euler1D (gamma=%f)" % self.gamma


class Euler2D:

    def __init__(self, gamma):
        self.gamma = gamma
        

        class U_vec(np.ndarray):
            """
                The vector of state, internally the values are stored in conservative form
            """
            num_components = 4
            @staticmethod
            def make(density, x_momentum = None, y_momentum = None, energy = None, x_velocity = None, y_velocity = None, pressure = None):
                vec = np.ndarray((4,)).view(U_vec)
                if not None in [density, x_momentum, y_momentum, energy]:
                    # Internally used variables
                    vec.density = density
                    vec.x_momentum = x_momentum
                    vec.y_momentum = y_momentum
                    vec.energy = energy
                elif not None in [density, x_velocity, y_velocity, pressure]:
                    # Calls to these variables are mapped internally
                    # NB! Order of these calls are important, e.g. need density to able to calculate velocity from momentum
                    vec.density = density
                    vec.set_x_velocity(x_velocity)
                    vec.set_y_velocity(y_velocity)
                    vec.set_pressure(pressure)
                elif density is None:
                    vec.density = 0.0
                    vec.x_momentum = 0.0
                    vec.y_momentum = 0.0
                    vec.energy = 0.0
                else:
                    raise TypeError("To define a state you must pass in at least (density,x_momentum,y_momentum,energy) or (density,x_velocity,y_velocity,pressure) using kwargs")
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

            density = property(conserved_var_getter(0), conserved_var_setter(0))
            x_momentum = property(conserved_var_getter(1), conserved_var_setter(1))
            y_momentum = property(conserved_var_getter(2), conserved_var_setter(2))
            energy = property(conserved_var_getter(3), conserved_var_setter(3))
            
            slopeCalcVar = property(conserved_var_getter(0))
            
            # Derived variables
            def get_x_velocity(self):
                return self.x_momentum/self.density
            def set_x_velocity(self, value):
                self.x_momentum = self.density*value
            
            def get_y_velocity(self):
                return self.y_momentum/self.density
            def set_y_velocity(self, value):
                self.y_momentum = self.density*value
            
            def get_pressure(self):
                return (gamma - 1.0)*(self.energy - 0.5*self.x_momentum**2/self.density - 0.5*self.y_momentum**2.0/self.density)
            def set_pressure(self, value):
                self.energy = 0.5*self.x_momentum**2/self.density + 0.5*self.y_momentum**2.0/self.density + value/(gamma-1.0)

            def get_internal_energy(self):
                return self.pressure/((gamma-1.0)*self.density)

            x_velocity = property(get_x_velocity, set_x_velocity)
            y_velocity = property(get_y_velocity, set_y_velocity)
            pressure = property(get_pressure, set_pressure)
            internal_energy = property(get_internal_energy)

            def getSoundSpeed(self):
                return np.sqrt(gamma*self.pressure/self.density)



            def debug(self):
                print "density=%f, x_velocity=%f, y_velocity=%f, pressure=%f, x_momentum=%f, y_momentum=%f, energy=%f" % (self.density, self.x_velocity, self.y_velocity, self.pressure, self.x_momentum, self.y_momentum, self.energy)

            #def __str__(self):
                #return "%f\t%f\t%f\t%f\t%f" % (self.density, self.x_velocity, self.y_velocity, self.pressure, self.energy)

            def getSlopeCalcVar(self):
                return self.density
            
        def getMaxWaveSpeed(Q):
            vec = Q.view(U_vec)
            c = vec.getSoundSpeed()
            return (np.max(np.abs(vec.x_velocity) + c), np.max(np.abs(vec.x_velocity) + c) )
            
        def F(Q, dx):
            #u1 = vec.x_momentum
            #u2 = vec.x_momentum*vec.x_velocity + vec.pressure
            #u3 = vec.x_momentum*vec.y_velocity
            #u4 = vec.x_velocity*(vec.energy+vec.pressure)
            #return U_vec(density = u1, x_momentum = u2, y_momentum = u3, energy = u4)

            vec = Q.view(U_vec)
            x_m = vec.x_momentum
            x_v = vec.x_velocity
            y_v = vec.y_velocity
            p = vec.pressure
            E = vec.energy

            u1 = x_m
            u2 = ne.evaluate("x_m*x_v + p")
            u3 = ne.evaluate("x_m*y_v")
            u4 = ne.evaluate("x_v*(E+p)")

            return np.dstack((u1, u2, u3, u4))
        
        def G(Q, dx):
            #u1 = vec.y_momentum
            #u2 = vec.y_momentum*vec.x_velocity
            #u3 = vec.y_momentum*vec.y_velocity + vec.pressure
            #u4 = vec.y_velocity*(vec.energy+vec.pressure)
            #return U_vec(density = u1, x_momentum = u2, y_momentum = u3, energy = u4)

            vec = Q.view(U_vec)
            y_m = vec.y_momentum
            x_v = vec.x_velocity
            y_v = vec.y_velocity
            p = vec.pressure
            E = vec.energy

            u1 = y_m
            u2 = ne.evaluate("y_m*x_v")
            u3 = ne.evaluate("y_m*y_v + p")
            u4 = ne.evaluate("y_v*(E+p)")
            return np.dstack((u1, u2, u3, u4))
        
        class EulerUtils:
            @staticmethod
            def getLeftStateForRightShock(w_right, S_3):
                """
                    Toro p. 101
                """
                (rho_r, u_r, p_r) = (w_right.density, w_right.velocity, w_right.pressure)

                a_r = sqrt(p_r*gamma/rho_r)
                M_r = u_r/a_r

                M_s = S_3/a_r

                rho_s = rho_r*((gamma+1.0)*(M_r-M_s)**2.0)/((gamma-1.0)*(M_r-M_s)**2.0+2.0)
                p_s = p_r*(2.0*gamma*(M_r-M_s)**2.0-(gamma-1.0))/(gamma+1.0)
                u_s = (1.0-rho_r/rho_s)*S_3 + u_r*rho_r/rho_s

                return U_vec(density=rho_s, velocity=u_s, pressure=p_s)

            @staticmethod
            def getRightStateForLeftShock(w_left, S_1):
                """
                    Toro p. 102
                """
                (rho_l, u_l, p_l) = (w_left.density, w_left.velocity, w_left.pressure)

                a_l = sqrt(p_l*gamma/rho_l)
                M_l = u_l/a_l

                M_s = S_1/a_l

                rho_s = rho_l*((gamma+1.0)*(M_l-M_s)**2.0)/((gamma-1.0)*(M_l-M_s)**2.0+2.0)
                p_s = p_l*(2.0*gamma*(M_l-M_s)**2.0-(gamma-1.0))/(gamma+1.0)
                u_s = (1.0-rho_l/rho_s)*S_1 + u_l*rho_l/rho_s
                
                return U_vec(density=rho_s, velocity=u_s, pressure=p_s)
        
        class Euler2DRiemannSolver:
            """
                Almost identical to Euler1DRiemannSolver, however including the transverse shear wave (as discussed in Toro p. 111).
                The jump in the transverse velocity only occours across this shear wave, the speed of the shear wave coincides with
                the contact discontinuity.
            """
            def __init__(self,q_l,q_r, norm):
                """
                    Initiate the 2D Riemann Problem solver. The direction of the solve is given by the norm tuple.

                    The y-direction solve is performed identically to the x-direction however with the x and y-velocity compontents swapped.
                """
                self.gamma = gamma

                self.state_l = q_l.view(U_vec)
                self.state_r = q_r.view(U_vec)
                self.norm = norm

                # Here it now becomes important in which direction we are solving the RiemannProblem
                (rho_l, p_l, rho_r, p_r) = (self.state_l.density, self.state_l.pressure, self.state_r.density, self.state_r.pressure)

                if norm == (1,0):
                    (u_l, u_r) = (self.state_l.x_velocity, self.state_r.x_velocity)
                    (v_l, v_r) = (self.state_l.y_velocity, self.state_r.y_velocity)
                elif norm == (0,1):
                    (u_l, u_r) = (self.state_l.y_velocity, self.state_r.y_velocity)
                    (v_l, v_r) = (self.state_l.x_velocity, self.state_r.x_velocity)

                a_l = self.soundSpeed(rho_l, p_l)
                a_r = self.soundSpeed(rho_r, p_r)

                (p_s, u_s) = self.__findPStarAndUStar()
                self.p_s = p_s
                self.u_s = u_s

                gammaRatio = (gamma-1.0)/(gamma+1.0)


                # Speed of contact wave 
                self.S_2 = u_s

                if self.hasLeftShock():
                    rho_s_l = rho_l*( ((p_s/p_l) + (gamma-1.0)/(gamma+1.0)) / ( ((gamma-1.0)/(gamma+1.0))*(p_s/p_l) + 1.0))
                    self.S_L = u_l - a_l*sqrt( (gamma+1.0)/(2.0*gamma)*(p_s/p_l) + (gamma-1.0)/(2.0*gamma) )
                else:
                    rho_s_l = rho_l*(p_s/p_l)**(1.0/gamma)
                    a_s_l = a_l * (p_s/p_l)**((gamma-1.0)/(2.0*gamma))
         
                    # Head of the left rarefaction
                    self.S_HL = u_l - a_l
                    # Tail of the left rarefaction 
                    self.S_TL = u_s - a_s_l
                
                if self.norm == (1,0):
                    self.state_l_s = U_vec.make(density=rho_s_l,pressure=p_s,x_velocity=u_s,y_velocity=v_l)
                elif self.norm == (0,1):
                    self.state_l_s = U_vec.make(density=rho_s_l,pressure=p_s,x_velocity=v_l,y_velocity=u_s)


                if self.hasRightShock():
                    rho_s_r = rho_r*( ((p_s/p_r) + (gamma-1.0)/(gamma+1.0)) / ( ((gamma-1.0)/(gamma+1.0))*(p_s/p_r) + 1.0))
                    self.S_R = u_r + a_r*sqrt( (gamma+1.0)/(2.0*gamma)*(p_s/p_r) + (gamma-1.0)/(2.0*gamma) )
                else:
                    rho_s_r = rho_r*(p_s/p_r)**(1.0/gamma)
                    a_s_r = a_r * (p_s/p_r)**((gamma-1.0)/(2.0*gamma))
                    # Head of the right rarefaction
                    self.S_HR = u_r + a_r
                    # Tail of the right rarefaction
                    self.S_TR = u_s + a_s_r
                
                if self.norm == (1,0):
                    self.state_r_s = U_vec.make(density=rho_s_r,pressure=p_s,x_velocity=u_s,y_velocity=v_r)
                elif self.norm == (0,1):
                    self.state_r_s = U_vec.make(density=rho_s_r,pressure=p_s,x_velocity=v_r,y_velocity=u_s)


            def getStates(self, dtdx = None):
                states = [self.state_l]

                if self.hasLeftShock():
                    states.append(self.state_l_s)
                else:
                    if self.leftRarefactionIsSonic():
                        # Add central state, time=0.1 is not important, just have to make sure we don't devide by zero in dx/dt
                        x = 1.0
                        t = x*dtdx
                        states.append(self.getState(x/t))
                    else:
                        states.append(self.state_l_s)

                if self.hasRightShock():
                    states.append(self.state_r_s)
                else:
                    if self.rightRarefactionIsSonic():
                        x = 1.0
                        t = x*dtdx
                        states.append(self.getState(x/t))
                    else:
                        states.append(self.state_r_s)

                states.append(self.state_r)
                
                return states
            
            def getWaveSpeeds(self):
                waveSpeeds = []

                if self.hasLeftShock():
                    waveSpeeds.append(self.S_L)
                else:
                    waveSpeeds.append(self.S_HL)

                # contact and shear wave move together
                waveSpeeds.append(self.S_2)
                waveSpeeds.append(self.S_2)

                if self.hasRightShock():
                    waveSpeeds.append(self.S_R)
                else:
                    waveSpeeds.append(self.S_HR)

                return waveSpeeds



            def soundSpeed(self,density,pressure):
                return sqrt(self.gamma*pressure/density)


            def __findPStarAndUStar(self):
                (rho_r, rho_l, p_r, p_l, gamma) = (self.state_r.density, self.state_l.density, self.state_r.pressure, self.state_l.pressure, self.gamma)
                
                if self.norm == (1,0):
                    (u_l, u_r) = (self.state_l.x_velocity, self.state_r.x_velocity)
                elif self.norm == (0,1):
                    (u_l, u_r) = (self.state_l.y_velocity, self.state_r.y_velocity)

                
                a_l = self.soundSpeed(rho_l, p_l)
                a_r = self.soundSpeed(rho_r, p_r)
                
                def A_k(rho_k):
                    return 2.0/((self.gamma + 1.0)*rho_k)

                def B_k(p_k):
                    return (gamma - 1.0)/(gamma + 1.0)*p_k

                def f_k(p, state_k):
                    """
                        Expression used to relate density and pressure in front of rarefaction/shock (state_k) to pressure inside (p)
                    """
                    (rho_k, p_k) = (state_k.density, state_k.pressure)
                    if p > p_k:
                        # shock
                        return (p-p_k)*sqrt(A_k(rho_k)/(p + B_k(p_k) ) )
                    else:
                        # rarefaction
                        return 2.0*self.soundSpeed(rho_k,p_k)/(gamma-1.0) * ( (p/p_k)**((gamma-1.0)/(2.0*gamma)) - 1.0 )

                def f(p):
                    """
                        Polynomial to solve, f=0, to find pressure in the star region
                    """
                    return f_k(p, self.state_l) + f_k(p, self.state_r) + u_r - u_l

                def dfdp(p):
                    f_d_value = 0.0
                    if p > p_l:
                        f_d_value += sqrt(A_k(rho_l)/(B_k(p_l)+p))*(1.0-(p-p_l)/(2.0*(B_k(p_l)+p)))
                    else:
                        f_d_value += 1.0/(rho_l*a_l)*(p/p_l)**(-(gamma+1.0)/(2.0*gamma))

                    if p > p_r:
                        f_d_value += sqrt(A_k(rho_r)/(B_k(p_r)+p))*(1.0-(p-p_r)/(2.0*(B_k(p_r)+p)))
                    else:
                        f_d_value += 1.0/(rho_r*a_r)*(p/p_r)**(-(gamma+1.0)/(2.0*gamma))
                    
                    return f_d_value


                def getUStar(p_s):
                    return 0.5*(u_l + u_r) + 0.5*(f_k(p_s, self.state_r) - f_k(p_s, self.state_l) )
                    
                

                k = (gamma-1.0)/(2.0*gamma)

                # double p_tr = pow((a_l + a_r - 0.5*(gamma-1.0)*(u_r-u_l))/(a_l/pow(p_l, (gamma-1.0)/(2.0*gamma)) + a_r/(pow(p_r, (gamma-1.0)/(2.0*gamma)))), 2*gamma/(gamma-1.0));
                p_tr = ( (a_l+a_r-0.5*(gamma-1.0)*(u_r-u_l))/(a_l/(p_l**k)+a_r/(p_r**k)) )**(1.0/k)
                p_s = rootFinder.NewtonRaphson().solve(epsilon, p_tr, f, dfdp)
                    
                return (p_s, getUStar(p_s))


            def hasLeftShock(self):
                return self.state_l.pressure < self.p_s
            
            def hasRightShock(self):
                return self.state_r.pressure < self.p_s

            def hasLeftRarefaction(self):
                return self.state_l.pressure > self.p_s
            
            def hasRightRarefaction(self):
                return self.state_r.pressure > self.p_s

            def leftRarefactionIsSonic(self):
                return self.hasLeftRarefaction() and self.S_HL < 0.0 and self.S_TL > 0.0

            def rightRarefactionIsSonic(self):
                return self.hasRightRarefaction() and self.S_TR < 0.0 and self.S_HR > 0.0

            def hasContactWave(self):
                return abs(self.state_l_s.density - self.state_r_s.density) > 2.0*epsilon
                

            def __str__(self):
                debugLines = []
                debugLines.append("%s, %s, %s, %s" % (self.state_l, self.state_l_s, self.state_r_s, self.state_r))
                if (self.hasLeftShock()):
                    debugLines.append("Left shock: \t\tS_L=%f" % self.S_L)
                elif self.hasLeftRarefaction():
                    debugLines.append("Left rarefaction: \t\tS_HL=%f, S_TL=%f" % (self.S_HL, self.S_TL))
                
                if self.hasContactWave():
                    debugLines.append("Contactwave: \t\tS_2=%f" % self.S_2)

                if self.hasRightShock():
                    debugLines.append("Right shock: \t\tS_R=%f" % self.S_R)
                elif self.hasRightRarefaction():
                    debugLines.append("Right rarefaction: \tS_TR=%f, S_HR=%f" % (self.S_TR, self.S_HR))
                 
                return "\n".join(debugLines)

            def getState(self, xt):
                dxdt = xt

                if dxdt < self.S_2:
                    if self.hasLeftShock():
                        if dxdt < self.S_L:
                            return self.state_l
                        else:
                            return self.state_l_s
                    else:
                        if dxdt < self.S_HL:
                            return self.state_l
                        elif dxdt < self.S_TL:
                            (rho_l, u_l, p_l, gamma) = (self.state_l.density, self.state_l.velocity, self.state_l.pressure, self.gamma)
                            a_l = self.soundSpeed(rho_l, p_l)
                            rho = rho_l*(2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*a_l)*(u_l-dxdt))**(2.0/(gamma-1.0))
                            u = 2.0/(gamma+1.0)*(a_l + (gamma-1.0)/2.0*u_l + dxdt)
                            p = p_l*(2.0/(gamma+1.0) + (gamma-1.0)/((gamma+1.0)*a_l)*(u_l - dxdt))**(2.0*gamma/(gamma-1.0))

                            if norm == (1,0):
                                return U_vec.make(density=rho, x_velocity=u, y_velocity=v_l, pressure=p)
                            elif norm == (0,1):
                                return U_vec.make(density=rho, x_velocity=v_l, y_velocity=u, pressure=p)
                        else:
                            return self.state_l_s
                else:
                    if self.hasRightShock():
                        if dxdt > self.S_R:
                            return self.state_r
                        else:
                            return self.state_r_s
                    else:
                        if dxdt > self.S_HR:
                            return self.state_r
                        elif dxdt > self.S_TR:
                            (rho_r, u_r, p_r, gamma) = (self.state_r.density, self.state_r.velocity, self.state_r.pressure, self.gamma)
                            a_r = self.soundSpeed(rho_r, p_r)
                            rho = rho_r*(2.0/(gamma+1.0)-(gamma-1.0)/((gamma+1.0)*a_r)*(u_r-dxdt))**(2.0/(gamma-1.0))
                            u = 2.0/(gamma+1.0)*(-a_r+(gamma-1.0)/2.0*u_r+dxdt)
                            p = p_r*(2.0/(gamma+1.0)-(gamma-1.0)/((gamma+1.0)*a_r)*(u_r-dxdt))**(2.0*gamma/(gamma-1.0))
                            if self.norm == (1,0):
                                return U_vec.make(density=rho,x_velocity=u, y_velocity=v_r,pressure=p)
                            elif self.norm == (0,1):
                                return U_vec.make(density=rho,x_velocity=v_r, y_velocity=u,pressure=p)
                            
                        else:
                            return self.state_r_s

        self.state = U_vec
        self.F = F
        self.G = G
        
        self.RiemannSolver = Euler2DRiemannSolver
        self.Utils = EulerUtils
        self.getMaxWaveSpeed = getMaxWaveSpeed

    def __str__(self):
        return "euler2D"
    



    




        
