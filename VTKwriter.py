import visit_writer
from numpy import pi, exp, sqrt
import numpy as np
import sys


def makePlumeModelVTKfile(pos, data, var_name, filename):
    """ Generates a vtk-file for the Gaussian plume model
    """
    # Output settings
    useBinary = True
    

    # Number of grid points
    if pos.ndim == 2:
        (NX, NY) = pos.shape
        (x_pos, y_pos) = pos
        NZ = 1
        z = [0.0]
    else:
        (NX, NY, NZ) = pos.shape
        (x_pos, y_pos, z_pos) = pos

    # [ name, dimension, centering, arrayOfValues ]
    vars = ((var_name, 1, 0, tuple(data)),)

    print "Writing..."
    visit_writer.WriteRectilinearMesh(filename, useBinary, tuple(x_pos), tuple(y_pos), tuple(z_pos), vars)

    print "Output written to", filename
