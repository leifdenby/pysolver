import numpy as np
import scipy.constants
from common import aslice

class IdealGas(object):
    """
        Class for describing general ideal gas.
    """
    def __init__(self, f, M):
        """
            f: degrees of freedom
            M: molar weight (g/mol)
        """
        self.f = f
        self.M = M

    def cv_m(self):
        """
        molar specific heat at constant volume
        """
        return self.f/2.0*scipy.constants.R

    def cp_m(self):
        """
        molar specific heat at constant pressure
        """
        return self.cv_m() + scipy.constants.R

    def cv(self):
        """
        mass specific heat at constant volume

        units of J/K/kg
        """
        return self.cv_m()/self.M*1000.0

    def cp(self):
        """
        mass specific heat at constant pressure

        units of J/K/kg
        """
        return self.cp_m()/self.M*1000.0

    def thermal_expansion(self, T = 298.0):
        return 1.0/T

    def gamma(self):
        return self.cp()/self.cv()

    def kappa(self):
        return scipy.constants.R/self.cp()

    def R(self):
        """
        Specific gas constant, in units of N*m/(kmol*K)
        """
        return scipy.constants.R/self.M*1000.0

    def __str__(self):
        return "Ideal gas (M=%f, f=%f)" % (self.M, self.f)

class DiatomicGas(IdealGas):
    def __init__(self, M):
        super(DiatomicGas, self).__init__(f=5.0, M=M)
    
    def __str__(self):
        return "Diatomic %s" % super(DiatomicGas, self).__str__()

class AtmosphericAir(DiatomicGas):
    def __init__(self):
        super(AtmosphericAir, self).__init__(M=28.97)

    def __str__(self):
        return "Atmospheric air (%s)" % super(AtmosphericAir, self).__str__()

