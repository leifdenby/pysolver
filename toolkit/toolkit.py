import numpy as np
import matplotlib.pyplot as plot

class TestWrapper:
    """
    TODO:
        - Should have a grid class that stores grid extent and discretization
        - In FV-case the exact function should be evaluated as cell averages, i.e. integrate over the cell
    """
    def __init__(self, test_rutine, exact_f, test_name):
        self.test_rutine = test_rutine
        self.exact_f = exact_f
        self.test_name = test_name

    def __call__(self, N):
        dx = 1.0/N
        dy = 1.0/N
        x = np.linspace(dx/2.0, 1.0-dx/2.0, N)
        y = np.linspace(dy/2.0, 1.0-dy/2.0, N)
        from common import meshgrid
        xx, yy = meshgrid(x, y)

        u_exact = (np.vectorize(self.exact_f))(xx, yy)
        (u_sim, sim_steps) = self.test_rutine(N=N)

        return (u_sim, u_exact, sim_steps)

def convergence_test(test_func, N_max=80, norm=2, save_png=False, save_data=False):
    results = []
    test_name = test_func.test_name
    print("Running %s" % test_name)
    
    N = 10
    while N < N_max:
        print "Conv. N=%i" % N

        (u, u_exact, sim_steps) = test_func(N=N)

        norm_err = np.linalg.norm(u-u_exact, ord=norm)
        
        results.append((N, norm_err/(N*N), sim_steps))
        N *= 2


    results = np.array(results)

    N = results[:,0]
    dx = 1.0/N
    err = results[:,1]
    sim_steps = results[:,2]

    (a, b) = np.polyfit(np.log(dx), np.log(err), 1)

    print "slope=%.3f" % a

    plot.ion()
    plot.plot(np.log(dx), np.log(err), '*', label="$L_{%s}$ norm" % str(norm))
    plot.plot(np.log(dx), b + a*np.log(dx), label="fit (slope=%.3f)" % a)
    plot.xlabel("ln(dx)")
    plot.ylabel("ln(err)")
    plot.legend(loc=4)
    plot.grid(True)
    
    if save_png:
        plot.savefig("convergence_test_%s.png" % (test_name), dpi=400)
    if save_data:
        filename = "convergence_test_%s.dat" % test_name
        file = open(filename,"w")
        file.write("#regression: %f x+%f\n" % (a, b))
        file.write("#dx, err, sim_steps\n")

        for i in range(len(err)):
            file.write("%f\t%f\t%i\n" % (dx[i], err[i], sim_steps[i]))
        file.close()

    plot.draw()
    raw_input()

