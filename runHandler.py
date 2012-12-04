"""
General purpose handler for a single run. Takes care of all the time keeping
and produces output regarding how a run is progressing

TODO:
    - Need to deal with output settings somehow too.
"""

import time
import numpy as np

def run(settings):
    method = settings.method

    if settings.interactive_settings is not None:
        (pause, output_every_n_steps, plotting_routine) = settings.interactive_settings

    walltime_start = time.time()
    output_times = settings.output_times
    t_end = output_times[-1]
    np.seterr(all='print')

    # the grid is defined specific to the method so that we have the right number of ghost cells
    grid = method.grid



    Q = grid.initiateCells(settings.test, settings.model)

    t = 0.0
    n_steps = 0

    print repr(settings.model)
    print grid

    while t_end is None and True or t < t_end:
        try:
            grid.applyBCs(Q=Q, boundary_conditions=settings.test.boundary_conditions)

            # timestep calculation
            dt = method.calculateTimestep(Q)
            if output_times is not None:
                if len(output_times) > 0 and t + dt > output_times[0]:
                    dt = output_times[0] - t
            walltime_duration = time.time() - walltime_start
            t += dt
            n_steps += 1
            print "[%ds] %i\tt = %f, dt = %f" % (walltime_duration, n_steps, t, dt)
            
            # Main loop of update in a single timestep here
            method.evolve(Q, dt)

            if settings.interactive_settings is not None:
                if (n_steps-1) % output_every_n_steps == 0 or len(output_times) > 0 and output_times[0] <= t:
                    if len(output_times) > 0 and output_times[0] == t:
                        output_times.pop(0)
                    plotting_routine(Q=Q, grid=grid, model=settings.model, test=settings.test, n_steps=n_steps, t=t, num_scheme=method)
                    if pause:
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            print
            print "Simulation paused."
            a = raw_input()
            if a == "w":
                pass
                #do_writeoutput()
            elif a == "q":
                quit()
            elif a == "n":
                break
            elif a == "p":
                plotting_routine(Q, grid, settings.model, grid.num_ghost_cells, settings.test, title = "Q_%d" % n_steps, n_steps=n_steps, t=t)
