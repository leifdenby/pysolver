import numpy as np
import scipy.constants
import atmosphericFlow.gasProperties

def getStandardIsothermalAtmosphere():
    gas_properties = atmosphericFlow.gasProperties.AtmosphericAir()
    rho0 = 1.205
    p0 = 101325.0
    dTdz = 0.0  # isothermal
    g = scipy.constants.g
    return HydrostaticallyBalancedAtmosphere(rho0=rho0, p0=p0, dTdz=dTdz, gas_properties=gas_properties, g=g)
    
def getStandardIsentropicAtmosphere(gas_properties):
    rho0 = 1.205
    p0 = 101325.0
    g = scipy.constants.g
    dTdz = -g/gas_properties.cp()
    return HydrostaticallyBalancedAtmosphere(rho0=rho0, p0=p0, dTdz=dTdz, gas_properties=gas_properties, g=g)

class HydrostaticallyBalancedAtmosphere:
    """
    Class for setting a hydrostatically balanced atmosphere with
    and ideal gas.

    p = rho*R/M*T

    R: Unified gas constant
    """
    def __init__(self, rho0, p0, dTdz, gas_properties, g = None):
        self.rho0 = rho0
        self.p0 = p0
        self.dTdz = dTdz
        self.gas_properties = gas_properties
        self.T0 = p0*gas_properties.M/(rho0*scipy.constants.R)
        if g is None:
            self.g = scipy.constants.g
        else:
            self.g = g

    def __str__(self):
        return "HydrostaticallyBalancedAtmosphere (rho0=%f, p0=%f, dTdz=%f) with %s" % (self.rho0, self.p0, self.dTdz, str(self.gas_properties))
    
    def temp(self, pos):
        z = pos[-1]
        return self.T0 + self.dTdz*z

    def rho(self, pos):
        z = pos[-1]
        if self.dTdz == 0.0:
            return self.rho0*np.exp(-z*self.g*self.gas_properties.M/(scipy.constants.R*self.T0))
        else:
            alpha = self.g*self.gas_properties.M/(self.dTdz*scipy.constants.R)
            return self.rho0*np.power(self.temp(z), -alpha - 1.0)

    def drho_dz(self, pos):
        z = pos[-1]
        if self.dTdz == 0.0:
            return -self.g*self.gas_properties.M/(scipy.constants.R*self.T0)*self.rho(pos)
        else:
            alpha = self.g*self.gas_properties.M/(self.dTdz*scipy.constants.R)
            return (-alpha-1.0)*np.power(self.temp(pos), -alpha - 2.0)*self.dTdz

    def p(self, pos):
        z = pos[-1]
        return self.rho0*scipy.constants.R/self.gas_properties.M*self.temp(z)

    def theta(self, pos):
        """
        Calculate the potential temperature at z.
        """
        z = pos[-1]
        return self.temp(z)*np.power(self.p(pos)/self.p0, self.gas_properties.kappa())

    def vel(self, pos):
        return np.zeros(pos.shape)
