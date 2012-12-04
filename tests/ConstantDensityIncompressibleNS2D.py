from grids import boundary_conditions as BCs
import models

import numpy as np

PROJECT_ROOT = '/Users/leifdenby/Desktop/PhD/low_Mach_code/'

class LidDrivenCavity():
    def __init__(self, domain_spec, u_top, viscosity):
        self.u_top = u_top
        self.boundary_conditions = { 
                0: (BCs.SolidWall(), BCs.SolidWall(), BCs.NoSlip(), BCs.MovingWall(wall_velocity=u_top)),
                1: (BCs.NoSlip(), BCs.NoSlip(), BCs.SolidWall(), BCs.SolidWall()) 
                }
        self.model = models.ConstantDensityIncompressibleNS2D(viscosity)

        # FIXME: Take grid aspect-ratio into account with discrete solutions
        self.domain_spec = domain_spec
        self.Re = 1.0*u_top*max(domain_spec.getLengthScales())/viscosity
        
        self.__loadDiscreteSolutionData()
    
        self.ic_func = [lambda (x, y): 0.0*x*y, lambda (x, y): 0.0*x*y]
        

    def __loadDiscreteSolutionData(self):
        Re_s = [100, 400, 1000, 320, 5000, 7500, 10000]

        if int(self.Re) in Re_s:
            dtype = [('grid_point','i'), ('y','f')]
            for Re in Re_s:
                dtype.append(('Re_%i' % Re, 'f'))

            raw_data_u = np.loadtxt(PROJECT_ROOT + 'lib/tests/data/incompressibleNS/ghia_data/u_vel.dat',dtype=dtype)
            
            dtype = [('grid_point','i'), ('x','f')]
            for Re in Re_s:
                dtype.append(('Re_%i' % Re, 'f'))
            raw_data_v = np.loadtxt(PROJECT_ROOT + 'lib/tests/data/incompressibleNS/ghia_data/v_vel.dat',dtype=dtype)

            self.discrete_solution = {
                    'u_vel' : { 'y' : raw_data_u['y'], 'u' : raw_data_u['Re_%i' % self.Re]},
                    'v_vel' : { 'x' : raw_data_v['x'], 'v' : raw_data_v['Re_%i' % self.Re]},
                    'source' : 'Ghia et al. 1982',
                    }


    def __str__(self):
        return "Lid-driven cavity problem for constant density NS (u_top=%f)" % self.u_top

