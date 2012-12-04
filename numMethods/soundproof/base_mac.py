"""
This is an implementation of the original MAC-method for solving the generalized low Mach
models. To conform with my general code layout a finite volume cell averaged grid domain is
used to communicate data values with the outside world. Internal velocities are stored at
cell edges in the traditional MAC-type staggered grid arrangement.

It uses
- pyamg multigrid solver (with the faster discrete_laplacian_operator)
- getPressureCoefficient from the model
- getLaplacianCoefficient from the model
- Godunov-type upwinding of the scalar variables as in the BCG method, using the MAC-projected interface velocities
- time-explicit evaluation of the viscous terms

TODO:
    - boundary-conditions are set explicitly on velocity, this is not a problem for regid wall problems
      but in future for outflow problems this will need to be changed
    - Include the divergence coefficient term in the projection step so that all low Mach models may be used

"""

import numpy as np
import scipy.sparse.linalg
from math import sqrt
import time
import pyamg

from common import print_grid, meshgrid
import grids.boundary_conditions as BCs
import poisson_solver.nonlinear_poisson as nonlinear_poisson

import models

from grids import grid2d

import warnings

epsilon = 1.0e-16

def run(model, test, domain_spec, t_end, plotting_routine, pause, output_times = None, output_every_n_steps=None):
    plotting = True
    num_ghost_cells = 1
    grid = grid2d.MAC(domain_spec, num_ghost_cells)
    grid_fv = grid2d.FV(domain_spec, num_ghost_cells)

    if type(model) is models.ConstantDensityIncompressibleNS2D:
        state_positions = [grid2d.StatePosition.interface_x, grid2d.StatePosition.interface_y]
    elif type(model) is models.VariableDensityIncompressibleNS2D:
        state_positions = [grid2d.StatePosition.interface_x, grid2d.StatePosition.interface_y, grid2d.StatePosition.center]
    else:
        raise Exception("The first-order MAC-type method does not know how to solve the (%s) model yet" % str(model))
    
    Q = grid_fv.initiateCells(test, model)


    warnings.warn("The selected numerical method uses a MAC-staggered grid internally, Q will only be updated when plotting is requested")
    u = grid_fv.makeInterfaceStates(Q, axis=0, boundary_conditions=test.boundary_conditions).view(model.state).x_velocity
    v = grid_fv.makeInterfaceStates(Q, axis=1, boundary_conditions=test.boundary_conditions).view(model.state).y_velocity


    (N_x, N_y) = domain_spec.N
    (dx, dy) = domain_spec.getGridSpacing()
    
    print repr(model)
    print "(dx, dy) = (%f, %f)" % (dx, dy)
    
    t = 0.0
    n_steps = 0
    start_time = time.time()

    cfl_adv = 0.9
    cfl_visc = 0.25
    
    plotting_routine(Q=Q, grid=grid_fv, model=model, test=test, n_steps=n_steps, t=t, num_scheme="MAC-method")
    raw_input()

    while t_end is None and True or t < t_end:
        try:
            n = num_ghost_cells
            (u_max, v_max) = (np.max(np.abs(u[n:-n,n:-n])), np.max(np.abs(v[n:-n,n:-n])))
            dt_constraints = []
            g = getattr(model, 'g', 0.0)
            if not g == 0.0:
                dt_constraints.append(cfl_adv*sqrt(2.0*dx/g))
            mu = getattr(model, 'viscosity', 0.0)
            if not mu == 0.0:
                if hasattr(model.state, 'rho'):
                    dt_constraints.append(cfl_visc*(min(dx,dy))**2.0*np.min(Q.view(model.state).rho[n:-n,n:-n])/mu)
                else:
                    dt_constraints.append(cfl_visc*(min(dx,dy))**2.0/mu)
            if not u_max == 0.0:
                dt_constraints.append(cfl_adv*dx/u_max)
            if not v_max == 0.0:
                dt_constraints.append(cfl_adv*dy/v_max)
            #for c, bcs in test.boundary_conditions.items():
                #for i, bc in enumerate(bcs):
                    #if isinstance(bc, BCs.MovingWall):
                        #axis = i / 2
                        #dx_ = grid.getGridSpacing()[axis]
                        #dt_constraints.append(cfl_adv*dx_/bc.wall_velocity)

            dt = np.min(dt_constraints)

            if output_times is not None:
                if len(output_times) > 0 and t + dt > output_times[0]:
                    dt = output_times[0] - t
            
            duration = time.time() - start_time
            t += dt
            print "[%ds] %i\tt = %f, dt = %f, u_max = %e, v_max = %e" % (duration, n_steps, t, dt, u_max, v_max)
            n_steps += 1

            # Apply BCs
            u[0,1:N_y+1]=0 # left wall
            u[N_x,1:N_y+1]=0 # right wall
            v[1:N_x+1,0]=0 # bottom wall
            v[1:N_x+1,N_y]=0 # top wall
            v[0,1:N_y+1]=2*0-v[1,1:N_y+1] # left ghost
            v[N_x+1,1:N_y+1]=2*0-v[N_x,1:N_y+1] # right ghost
            u[0:N_x+2,0]=2*0-u[0:N_x+2,1] # bottom ghost
            top_bc = test.boundary_conditions[0][3]
            if isinstance(top_bc, BCs.MovingWall):
                u[0:N_x+2,N_y+1]=2*top_bc.wall_velocity-u[0:N_x+2,N_y] # top ghost

            # Calculate U*
            u_star = np.zeros((N_x+1,N_y+2))
            v_star = np.zeros((N_x+2,N_y+1))
            
            usqr_1=(0.5*(u[1+1:N_x+1,1:N_y+1]+u[1:N_x,1:N_y+1]))**2
            usqr_2=(0.5*(u[1-1:N_x-1,1:N_y+1]+u[1:N_x,1:N_y+1]))**2
            uv_x1=0.5*(u[1:N_x,1:N_y+1]+u[1:N_x,1+1:N_y+1+1])*0.5*(v[1:N_x,1:N_y+1]+v[1+1:N_x+1,1:N_y+1])
            uv_x2=0.5*(u[1:N_x,1:N_y+1]+u[1:N_x,1-1:N_y+1-1])*0.5*(v[1:N_x,1-1:N_y+1-1]+v[1+1:N_x+1,1-1:N_y+1-1])
            # Advection term X-direction
            A=(usqr_1-usqr_2)/dx+(uv_x1-uv_x2)/dy
            # Laplacian operator for u
            D2_u=((u[1+1:N_x+1,1:N_y+1]-2*u[1:N_x,1:N_y+1]+u[1-1:N_x-1,1:N_y+1])/dx**2)+((u[1:N_x,1+1:N_y+1+1]-2*u[1:N_x,1:N_y+1]+u[1:N_x,1-1:N_y+1-1])/dy**2)
            if hasattr(model.state, 'rho'):
                visc_term_x = mu/(0.5*(Q.view(model.state).rho[1:N_x,1:N_y+1]+Q.view(model.state).rho[1+1:N_x+1,1:N_y+1]))*D2_u
            else:
                warnings.warn("The state-vector does not contain density, assuming rho=1.0 for now")
                visc_term_x = mu/1.0*D2_u
            u_star[1:N_x,1:N_y+1]=u[1:N_x,1:N_y+1]+dt*(-A+visc_term_x)

            vsqr_1=(0.5*(v[1:N_x+1,1+1:N_y+1]+v[1:N_x+1,1:N_y]))**2
            vsqr_2=(0.5*(v[1:N_x+1,1:N_y]+v[1:N_x+1,1-1:N_y-1]))**2
            uv_y1=0.5*(u[1:N_x+1,1+1:N_y+1]+u[1:N_x+1,1:N_y])*0.5*(v[1:N_x+1,1:N_y]+v[1+1:N_x+1+1,1:N_y])
            uv_y2=0.5*(u[1-1:N_x+1-1,1+1:N_y+1]+u[1-1:N_x+1-1,1:N_y])*0.5*(v[1:N_x+1,1:N_y]+v[1-1:N_x+1-1,1:N_y])
            # Advection term Y-direction
            B=(uv_y1-uv_y2)/dx+(vsqr_1-vsqr_2)/dy
            # Laplacian operator for v
            D2_v=((v[1+1:N_x+1+1,1:N_y]-2*v[1:N_x+1,1:N_y]+v[1-1:N_x+1-1,1:N_y])/dx**2)+((v[1:N_x+1,1+1:N_y+1]-2*v[1:N_x+1,1:N_y]+v[1:N_x+1,1-1:N_y-1])/dy**2)

            pos = grid_fv.getInterfacePositions(axis=1)
            buoyancy_terms = model.calculateBuoyancyTerm(Q=grid_fv.makeInterfaceStates(Q, axis=1, boundary_conditions=test.boundary_conditions), pos=pos)

            if hasattr(model.state, 'rho'):
                visc_term_y = mu/(0.5*(Q.view(model.state).rho[1:N_x+1,1:N_y]+Q.view(model.state).rho[1:N_x+1,1+1:N_y+1]))*D2_v
            else:
                visc_term_y = mu/1.0*D2_v

            v_star[1:N_x+1,1:N_y]=v[1:N_x+1,1:N_y]+dt*(-B+visc_term_y+buoyancy_terms.view(model.state).y_velocity[n:-n,n:-n])


            # Do projection on U*
            s = 1.0/dt*( (u_star[1:,1:-1] - u_star[:-1,1:-1])/dx + (v_star[1:-1,1:] - v_star[1:-1,:-1])/dy )

            #a_x = np.ones((N_x+1,N_y))/(dx**2.0)
            #a_y = np.ones((N_x,N_y+1))/(dy**2.0)
            a_x = model.getLaplacianCoefficients(Q, axis=0, grid=grid_fv, boundary_conditions=test.boundary_conditions)/(dx*dx)
            a_y = model.getLaplacianCoefficients(Q, axis=1, grid=grid_fv, boundary_conditions=test.boundary_conditions)/(dy*dy)

            a = (a_x, a_y)
            bcs = (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())

            #D_p = nonlinear_poisson.discrete_nonlinear_laplacian_2d(domain_spec.N, a, bcs) 
            D_p = nonlinear_poisson.new_discrete_nonlinear_laplacian_2d(domain_spec.N, a, bcs) 


            #(p_new, info) = scipy.sparse.linalg.cg(D_p, s)
            ml_p = pyamg.ruge_stuben_solver(D_p)
            p_new = ml_p.solve(s.ravel(), tol=epsilon)

            p_new = np.resize(p_new, (N_x, N_y))
            
            
            # Update velocity field
            
            u[n:-n,n:-n] = u_star[n:-n,n:-n] - dt*(p_new[1:,:]-p_new[:-1,:])/dx*model.getPressureCoefficients(Q=Q, axis=0, grid=grid_fv, boundary_conditions=test.boundary_conditions)[n:-n,:]
            v[n:-n,n:-n] = v_star[n:-n,n:-n] - dt*(p_new[:,1:]-p_new[:,:-1])/dy*model.getPressureCoefficients(Q=Q, axis=1, grid=grid_fv, boundary_conditions=test.boundary_conditions)[:,n:-n]
                    

            # Now that we have the velocity at the cell interfaces we need to advect the scalar fields,
            # we will use a specialised form of the corner-transport upwind method
            
            # The interface states in the x-direction
            Q_t_x = upwind_states(Q_l=Q[:-1,n:-n],Q_r=Q[1:,n:-n],u_c=u[:,n:-n])
            # y-direction
            Q_t_y = upwind_states(Q_l=Q[n:-n,:-1],Q_r=Q[n:-n,1:],u_c=v[n:-n])

            # need to stack a number of vectors, so that the maths works out (really need a better way to do this)
            u_adv_x_s = stack_multiple_copies(u[:,n:-n], copies=model.state.num_components)
            v_adv_y_s = stack_multiple_copies(v[n:-n], copies=model.state.num_components)

            #conv_Q = (Q_t_x[1:,:] - Q_t_x[:-1,:])*0.5*(u_adv_x_s[1:,:]+u_adv_x_s[:-1,:])/dx + (Q_t_y[:,1:] - Q_t_y[:,:-1])*0.5*(v_adv_y_s[:,1:]+v_adv_y_s[:,:-1])/dy
            conv_Q = (Q_t_x[1:,:]*u_adv_x_s[1:,:] - Q_t_x[:-1,:]*u_adv_x_s[:-1,:])/dx + (Q_t_y[:,1:]*v_adv_y_s[:,1:] - Q_t_y[:,:-1]*v_adv_y_s[:,:-1])/dy

            for i in range(2, model.state.num_components):
                Q[...,i][n:-n,n:-n] = Q[n:-n,n:-n,i] + dt*(-conv_Q[...,i])
            grid_fv.applyBCs(Q, test.boundary_conditions)

            ## Advection of scalar fields complete
            
            if (n_steps-1) % output_every_n_steps == 0 or len(output_times) > 0 and output_times[0] <= t:
                if len(output_times) > 0 and output_times[0] == t:
                    output_times.pop(0)
                if plotting:
                    Q.view(model.state).x_velocity[n:-n,n:-n] = grid.makeCellCenteredStates(u, axis=0, boundary_conditions=test.boundary_conditions)[:,n:-n]
                    Q.view(model.state).y_velocity[n:-n,n:-n] = grid.makeCellCenteredStates(v, axis=1, boundary_conditions=test.boundary_conditions)[n:-n,:]
                    plotting_routine(Q=Q, grid=grid_fv, model=model, test=test, num_scheme = "MAC-method", n_steps=n_steps, t=t)
                else:
                    pass
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
                plotting_routine(Q, grid, model, num_ghost_cells, test, title = "Q_%d" % n_steps, n_steps=n_steps, t=t)

def upwind_states(Q_l, Q_r, u_c):
    """
    Picks between the left and right states depending on the sign of u_c.
    All three states vectors must have the same shape
    """
    Q_int_l_i = u_c > epsilon# 0.0
    Q_int_r_i = u_c < -epsilon#0.0
    Q_int = 0.5*(Q_l + Q_r)
    Q_int[Q_int_l_i] = Q_l[Q_int_l_i]
    Q_int[Q_int_r_i] = Q_r[Q_int_r_i]

    #plottingHelper = PlottingHelper()
    #plot = plottingHelper.get_plotter()
    #print_grid(u_c)
    #plot.imshow(np.rot90(Q_int_l_i), interpolation="nearest")
    #raw_input()
    return Q_int

def stack_multiple_copies(v, copies=3):
    (nx, ny) = v.shape
    Q = np.empty((nx, ny, copies))
    for i in range(copies):
        Q[...,i] = v
    return Q
