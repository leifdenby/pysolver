import numpy as np

def print_grid(u):
    (i_max, j_max) = u.shape
    print u.shape
    for i in range(i_max):
        for j in range(j_max):
            if u[i,j] == 0.0: # introduced because -0.0 is a number, and it would print wrong
                print "  0.0    ",
            else:
                if u[i,j] > 0.0:
                    print " ",
                else:
                    print "",
                print "%.3e" % u[i,j],
        print


class Domain2D:
    def __init__(self, limits, Ns):
        ((self.xmin, self.xmax), (self.ymin, self.ymax)) = limits
        self.Nx, self.Ny = Ns

    def meshgrid(self):
        return np.meshgrid(np.linspace(self.xmin, self.xmax, self.Nx), np.linspace(self.ymin, self.ymax, self.Ny))


    def dx(self):
        return ( (self.xmax-self.xmin)/self.Nx, (self.ymax-self.ymin)/self.Ny)


def meshgrid(x, y):
    xx, yy = np.meshgrid(x, y)
    return np.array([xx.T, yy.T])

# http://www.mail-archive.com/numpy-discussion@scipy.org/msg36672.html
# create a slice along a specific axis, using aslice(axis, start, end)
aslice = lambda axis, s, e: (slice(None),) * axis + (slice(s, e),)
