"""
    Cell-centered 2D FV model for solving general low Mach number models of the
    form

    u_t = -U.div(u) - a*P_x
    v_t = -U.div(v) - a*P_y + b
    T_t = -U.div(T)
    div(c*U) = 0

    U: 2D velocity field, T: temperature-like variable, P: pressure-like
    variable


    Unsplit 2nd-order Godunov-type numerical scheme by Bell, Colella and 
    Howell 1991 (based on BCG 1989) is used.
"""

from grids import grid2d, boundary_conditions as BCs
from common import print_grid, aslice
from poisson_solver.nonlinear_poisson import new_discrete_nonlinear_laplacian_2d as discrete_nonlinear_laplacian_2d

import numpy as np
import pyamg

import scipy.sparse

import time

class BCG:
    def __init__(self, model, boundary_conditions, domain_spec):
        self.model = model
        self.boundary_conditions = boundary_conditions
        (Nx, Ny) = domain_spec.getNumCells()

        self.boundary_conditions_pressure = (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())
        self.grad_p = np.zeros((Nx,Ny,model.state.num_components))
        
        self.cfl_adv = 0.9
        self.num_ghost_cells = 2
        self.tol = 1.0e-16
        self.epsilon = 1.0e-16
        
        self.grid = grid2d.FV(domain_spec, self.num_ghost_cells)

    def calculateTimestep(self, Q):
        model = self.model
        (dx, dy) = self.grid.getGridSpacing()
        n = self.grid.num_ghost_cells

        (u_max, v_max) = (np.max(np.abs(Q.view(model.state).x_velocity[n:-n,n:-n])), 
                            np.max(np.abs(Q.view(model.state).y_velocity[n:-n,n:-n])))
        dt_constraints = []
        g = getattr(model, 'g', 0.0)
        if not g == 0.0:
            dt_constraints.append(self.cfl_adv*np.sqrt(2.0*dx/g))
        #if not mu == 0.0:
            #dt_constraints.append(cfl_visc*(min(dx,dy))**2.0*np.min(rho)/mu)
        if not u_max == 0.0:
            dt_constraints.append(self.cfl_adv*dx/u_max)
        if not v_max == 0.0:
            dt_constraints.append(self.cfl_adv*dy/v_max)
        for c, bcs in self.boundary_conditions.items():
            for i, bc in enumerate(bcs):
                if isinstance(bc, BCs.MovingWall):
                    axis = i / 2
                    dx_ = self.grid.getGridSpacing()[axis]
                    dt_constraints.append(self.cfl_adv*dx_/bc.wall_velocity)

        dt = np.min(dt_constraints)

        return dt

    def evolve(self, Q, dt):
        model = self.model
        grid = self.grid
        N = grid.getNumCells()
        num_ghost_cells = grid.num_ghost_cells
        n = num_ghost_cells
        (Nx, Ny) = N
        (dx, dy) = grid.getGridSpacing()

        # 1.    Form divergence-free velocity field at cell edges.
        #       This is done through calculating terms in a cell-centered Taylor expansion

        # 1.a       Extrapolate interface states which include only contributions from normal derivatives
        #           (denoted by Q_h for "Q hat")

        (Q_h_l, Q_h_r) = calc_hat_states(Q=Q, dt=dt, dx=dx, model=model, axis=0, grid=grid)
        (Q_h_b, Q_h_t) = calc_hat_states(Q=Q, dt=dt, dx=dy, model=model, axis=1, grid=grid)



        # 1.b       Solve RP at interfaces
        Q_h_int_x = calc_RP(Q_h_l, Q_h_r, axis=0, model=model)
        Q_h_int_y = calc_RP(Q_h_b, Q_h_t, axis=1, model=model)

        
        # 1.b.i     The hat-states have only been found on the interfaces interior to the domain, for the
        #           derivative approximation below we need the boundary states too, fortunately we know
        #           exactly what the velocity should be on the boundary. (Use subscript b to denote that the boundary is included)

        Q_h_int_x_b = np.zeros((Nx+1,Ny, model.state.num_components))
        Q_h_int_x_b[1:-1,...] = Q_h_int_x
        Q_h_int_y_b = np.zeros((Nx,Ny+1, model.state.num_components))
        Q_h_int_y_b[:,1:-1] = Q_h_int_y

        #Q_h_int_y_b[:,0,2] = Q_h_int_y_b[:,1,2] #+ 0.001*dy
        #Q_h_int_y_b[:,-1,2] = Q_h_int_y_b[:,-2,2] #- 0.001*dy
        #Q_h_int_x_b[0,:,2] = Q_h_int_x_b[1,:,2]
        #Q_h_int_x_b[-1,:,2] = Q_h_int_x_b[-2,:,2]

        # 1.c       Now that we have calculated the hat-states at the interfaces, we may now approximate the transverse derivatives
        # x_velocity from RPs averaged in x-direction
        u_h_int_x = Q_h_int_x_b.view(model.state).x_velocity
        u_x_avg = 0.5*(u_h_int_x[1:,...] + u_h_int_x[:-1,...])
        # y_velocity from RPs averaged in y-direction
        v_h_int_y = Q_h_int_y_b.view(model.state).y_velocity
        v_y_avg = 0.5*(v_h_int_y[:,1:,...] + v_h_int_y[:,:-1,...])
        
        # each component is multiplied by velocity, so stack'em
        u_x_avg_n = stack_multiple_copies(u_x_avg, model.state.num_components)
        udQdx = u_x_avg_n*(Q_h_int_x_b[1:,...] - Q_h_int_x_b[:-1,...])/dx
        # each component is multiplied by velocity, so stack'em
        v_y_avg_n = stack_multiple_copies(v_y_avg, model.state.num_components)
        vdQdy = v_y_avg_n*(Q_h_int_y_b[:,1:,...] - Q_h_int_y_b[:,:-1,...])/dy

        # 1.d       Calculate the bar states
        # first need the Laplacian of the velocity field (for the dispersion terms)
        LQ = calc_fivepoint_laplacian(Q, dx, dy)[1:-1,1:-1]

        pos = grid.getCellCenterPositions(n=num_ghost_cells)
        source_term = dt*0.5*model.calculateBuoyancyTerm(Q=Q, pos=pos)
        diffusion_coeffs = model.getDiffusionCoeffs()
        
        Q_bar_l = Q_h_l - dt/2.0*vdQdy[:-1,...]   + dt/2.0*diffusion_coeffs*LQ[:-1,...]     + source_term[2:-n-1,n:-n]
        Q_bar_r = Q_h_r - dt/2.0*vdQdy[1:,...]    + dt/2.0*diffusion_coeffs*LQ[1:,...]      + source_term[3:-n,n:-n]
        Q_bar_b = Q_h_b - dt/2.0*udQdx[:,:-1,...] + dt/2.0*diffusion_coeffs*LQ[:,:-1,...]   + source_term[n:-n,2:-n-1] 
        Q_bar_t = Q_h_t - dt/2.0*udQdx[:,1:,...]  + dt/2.0*diffusion_coeffs*LQ[:,1:,...]    + source_term[n:-n,3:-n]


        # 1.e       solve RPs again, for the bar states, same way as above
        Q_bar_int_x = calc_RP(Q_bar_l, Q_bar_r, axis=0, model=model)
        Q_bar_int_y = calc_RP(Q_bar_b, Q_bar_t, axis=1, model=model)
        
        # 1.e.i     Again, we need to add the boundary states, just velocity for now, scalars will be interesting
        Q_bar_int_x_b = np.zeros((Nx+1,Ny, model.state.num_components))
        Q_bar_int_x_b[1:-1,:] = Q_bar_int_x

        Q_bar_int_y_b = np.zeros((Nx,Ny+1, model.state.num_components))
        #Q_bar_int_y_b.view(model.state).x_velocity[:,-1] = 1.0
        Q_bar_int_y_b[:,1:-1] = Q_bar_int_y
        

        # 1.f       we now have the bar-states, time to make a MAC-projection!
        u_bar_int_x = Q_bar_int_x_b.view(model.state).x_velocity
        v_bar_int_y = Q_bar_int_y_b.view(model.state).y_velocity
        u_bar_avg_x = 0.5*(u_bar_int_x[1:,:]+u_bar_int_x[:-1,:])
        v_bar_avg_y = 0.5*(v_bar_int_y[:,1:]+v_bar_int_y[:,:-1])
        
        # 1.f.i     calc the MAC-projection source-term (still following suggestions in BCG 1999)

        #print "## FIRST PROJECTION"

        u_bar_int_x_b = Q_bar_int_x_b.view(model.state).x_velocity
        v_bar_int_y_b = Q_bar_int_y_b.view(model.state).y_velocity
        
        pos0 = grid.getInterfacePositions(axis=0)
        pos1 = grid.getInterfacePositions(axis=1)
        (c_x, c_y) = (
                        model.getDivergenceCoefficients(pos=pos0),
                        model.getDivergenceCoefficients(pos=pos1),
                    )

        cu_bar = u_bar_int_x_b*c_x
        cv_bar = v_bar_int_y_b*c_y
        
        s = 2.0/dt*((cu_bar[1:,...]-cu_bar[:-1,...])/dx + (cv_bar[:,1:,...] - cv_bar[:,:-1,...])/dy)
        #div_u_pre_mac = ((u_bar_int_x_b[1:,...]-u_bar_int_x_b[:-1,...])/dx + (v_bar_int_y_b[:,1:,...] - v_bar_int_y_b[:,:-1,...])/dy)


        (a_x, a_y) = (
                        model.getLaplacianCoefficients(Q=Q, axis=0, grid=grid, boundary_conditions=self.boundary_conditions)/(dx*dx),
                        model.getLaplacianCoefficients(Q=Q, axis=1, grid=grid, boundary_conditions=self.boundary_conditions)/(dy*dy),
                        )
        a = (a_x, a_y)

        D_p = discrete_nonlinear_laplacian_2d(N, a, self.boundary_conditions_pressure)
        # initiate multigrid solver
        ml_p = pyamg.ruge_stuben_solver(D_p)
        p_new = ml_p.solve(s.ravel(), tol=self.tol)
        #p_new = spsolve(D2, s)
        #(p_new, info) = scipy.sparse.linalg.cg(D_p, s.ravel(), tol=1.0e-20)
        

        p_new = np.resize(p_new, N)



        # 1.f.ii    Now to correct the interface velocities, for this we will need the pressure gradients at interfaces
        #           and we further require the pressure in the first row of ghost cells (to calculate the transverse gradients)
        # (this should actually have the ambient stratification in setting the boundary states I think, I need to look at this
        # later)
        p = np.ones((Nx+2,Ny+2))
        p[1:-1,1:-1] = p_new
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
        p[:,0] = p[:,1]
        p[:,-1] = p[:,-2]

        dpdx_x = 1.0/dx*(p[1:,1:-1] - p[:-1,1:-1])
        dpdy_x = 1.0/(4.0*dy)*(p[:-1,2:]+p[1:,2:] - (p[:-1,:-2]+p[1:,:-2]))
        dpdx_y = 1.0/(4.0*dx)*(p[2:,:-1]+p[2:,1:] - (p[:-2,:-1]+p[:-2,1:]))
        dpdy_y = 1.0/dy*(p[1:-1,1:] - p[1:-1,:-1])

        Q_bar_int_x_b.view(model.state).x_velocity -= dt/2.0*dpdx_x*model.getPressureCoefficients(Q=Q, grid=grid, axis=0, boundary_conditions=self.boundary_conditions)
        Q_bar_int_x_b.view(model.state).y_velocity -= dt/2.0*dpdy_x*model.getPressureCoefficients(Q=Q, grid=grid, axis=0, boundary_conditions=self.boundary_conditions)
        Q_bar_int_y_b.view(model.state).x_velocity -= dt/2.0*dpdx_y*model.getPressureCoefficients(Q=Q, grid=grid, axis=1, boundary_conditions=self.boundary_conditions)
        Q_bar_int_y_b.view(model.state).y_velocity -= dt/2.0*dpdy_y*model.getPressureCoefficients(Q=Q, grid=grid, axis=1, boundary_conditions=self.boundary_conditions)
       
        
        # Setting the boundary states in the following manner is not very neat, ideally I would have general way of 
        # dealing with this. I could also just not apply the projection
        Q_bar_int_x_b.view(model.state).x_velocity[-1,:] = 0.0
        Q_bar_int_x_b.view(model.state).x_velocity[0,:] = 0.0
        Q_bar_int_y_b.view(model.state).x_velocity[:,-1] = 0.0
        Q_bar_int_y_b.view(model.state).x_velocity[:,0] = 0.0
        Q_bar_int_x_b.view(model.state).y_velocity[-1,:] = 0.0
        Q_bar_int_x_b.view(model.state).y_velocity[0,:] = 0.0
        Q_bar_int_y_b.view(model.state).y_velocity[:,-1] = 0.0
        Q_bar_int_y_b.view(model.state).y_velocity[:,0] = 0.0
        #Q_bar_int_y_b[:,0,2] = Q_h_int_y_b[:,1,2] #+ 0.0001*dy
        #Q_bar_int_y_b[:,-1,2] = Q_h_int_y_b[:,-2,2] #- 0.0001*dy
        #Q_bar_int_x_b[0,:,2] = Q_h_int_x_b[1,:,2]
        #Q_bar_int_x_b[-1,:,2] = Q_h_int_x_b[-2,:,2]
        
        # all done! We have time-centered states at the interfaces that satisfies the divergence constraint.
        # Actually, to follow Almgren et al. 1998 only the velocities of these interface states should be used
        # These will be used for upwinding all the states

        # 1.g   Determine the half-time interface states by upwinding, choosing between the cell-centered states by using the Q_bar velocities determined above
        #       In the notation of Almgren 1998 these will be the tilde (subscript t) states
        
        #print "s"
        #print_grid(s)
        #u_bar_int_x_b = Q_bar_int_x_b.view(model.state).x_velocity
        #v_bar_int_y_b = Q_bar_int_y_b.view(model.state).y_velocity


        u_adv_x = Q_bar_int_x_b.view(model.state).x_velocity
        v_adv_y = Q_bar_int_y_b.view(model.state).y_velocity
        s = 2.0/dt*((u_bar_int_x_b[1:,...]-u_bar_int_x_b[:-1,...])/dx + (v_bar_int_y_b[:,1:,...] - v_bar_int_y_b[:,:-1,...])/dy)

        #print "s(postMAC)"
        #print_asymmetry(s, sign=-1)
        #print "u_adv_x(postMAC)"
        #print_asymmetry(u_adv_x)
        #print "v_adv_y(postMAC)"
        #print_asymmetry(v_adv_y, sign=-1)
        
        #plot.plot(y[N/2,:], -dt*conv_Q.view(model.state).rho[N/2,:], marker="x", label="conv_Q.rho")
        #plot.plot(y[N/2,:], Q.view(model.state).rho[N/2,n:-n], marker="x", label="Q.rho")
        #plot.plot(y[N/2,:], Q.view(model.state).y_velocity[N/2,n:-n], marker="x", label="Q.y")
        #plot.figure(0)
        #plottingHelper.quiver(x, y, 0.5*(u_adv_x[1:]+u_adv_x[:-1]), 0.5*(v_adv_y[:,1:] + v_adv_y[:,:-1]), color="green", label="U_avg(postMAC)")
        #plot.imshow(np.rot90(s), interpolation="nearest", extent=grid.getExtent())
        #plot.grid()
        #plot.legend()
        #plot.draw()
        #raw_input()
        
        # The interface states in the x-direction
        Q_t_x = upwind_states(Q_l=Q[1:-2,n:-n],Q_r=Q[2:-1,n:-n],u_c=u_adv_x)
        # y-direction
        Q_t_y = upwind_states(Q_l=Q[n:-n,1:-2],Q_r=Q[n:-n,2:-1],u_c=v_adv_y)


        # need to stack a number of vectors, so that the maths works out (really need a better way to do this)
        u_adv_x_s = stack_multiple_copies(u_adv_x, copies=model.state.num_components)
        v_adv_y_s = stack_multiple_copies(v_adv_y, copies=model.state.num_components)

        #print Q_t_x.shape, u_adv_x_s.shape
        #conv_Q = (Q_t_x[1:,:]*u_adv_x_s[1:,:] - Q_t_x[:-1,:]*u_adv_x_s[:-1,:])/dx + (Q_t_y[:,1:]*v_adv_y_s[:,1:] - Q_t_y[:,:-1]*v_adv_y_s[:,:-1])/dy
        conv_Q = (Q_t_x[1:,:] - Q_t_x[:-1,:])*0.5*(u_adv_x_s[1:,:]+u_adv_x_s[:-1,:])/dx + (Q_t_y[:,1:] - Q_t_y[:,:-1])*0.5*(v_adv_y_s[:,1:]+v_adv_y_s[:,:-1])/dy


        # 2.    Now to solve for the dispersion terms (parabolic equation), we do this in an implicit manner for
        #       stability.
        
        Q2_star = np.zeros(Q.shape)
        Q2_star[...,2][n:-n,n:-n] = Q[n:-n,n:-n,2] + dt*(-conv_Q[...,2])
        Q_h = 0.5*(Q + Q2_star)

        pos = grid.getCellCenterPositions(n=0)
        buoyancy_terms = model.calculateBuoyancyTerm(Q=Q_h[n:-n,n:-n], pos=pos)
        
        grad_p_coeff = model.getPressureCoefficients(Q=Q, grid=grid, boundary_conditions=self.boundary_conditions)
        
        #bouyancy_terms = np.zeros(Q.shape)[n:-n,n:-n]
        
        Q_star = np.zeros(Q.shape)
        for i, diffusion_coefficient in enumerate(model.getDiffusionCoeffs()):
            if diffusion_coefficient == 0.0:
                #print "explicit"
                # do explicit evaluation of the star state for this component
                import warnings
                warnings.warn("2nd-order pressure correction is not enabled as it currently is unstable")
                #Q_star[...,i][n:-n,n:-n] = Q[n:-n,n:-n,i] + dt*(-conv_Q[...,i] + buoyancy_terms[...,i])
                #Q_star[...,i][n:-n,n:-n] = Q[n:-n,n:-n,i] + dt*(-conv_Q[...,i] + buoyancy_terms[...,i] + grad_p_coeff[...,i]*grad_p[...,i])
                Q_star[...,i][n:-n,n:-n] = Q[n:-n,n:-n,i] + dt*(-conv_Q[...,i] + buoyancy_terms[...,i] + self.grad_p[...,i])
                #Q_star[...,i][n:-n,n:-n] = Q[n:-n,n:-n,i] + dt*(-conv_Q[...,i])
            else:
                #print "implicit"
                # Solve general parabolic equation implicitly using Crank-Nicholson
                bcs = self.boundary_conditions[i]

                # first the Laplacian operator, augmented as this is a parabolic equation
                a = ( grid.getInterfacePositions(axis=0)[0]*0.0+1.0/(dx*dx), grid.getInterfacePositions(axis=1)[0]*0.0+1.0/(dy*dy))
                D_d = discrete_nonlinear_laplacian_2d((N, N), a, bcs)
                Dk = D_d - 2.0/(dt*diffusion_coefficient)*scipy.sparse.identity(N*N)
                ml_diff = pyamg.ruge_stuben_solver(Dk)

                # assemble the source-term
                s = 2.0/diffusion_coefficient*(0.0*self.grad_p[...,i] + conv_Q[...,i] - 2.0/dt*Q[n:-n,n:-n,i] - buoyancy_terms[...,i])

                # apply boundary-conditions to source term
                for j, bc in enumerate(bcs):
                    if isinstance(bc, BCs.Dirichlet):
                        axis = j / 2
                        b_slice = grid.sliceFromEdge(edge_i=j,row_num=0)
                        dx_ = grid.getGridSpacing()[axis]
                        s[b_slice] -= 4.0/(dx_*dx_)*bc.fixed_value
                    else:
                        raise Exception("General parabolic solver has only been implemented to handle Dirichlet BCs.")
            
                r_star = ml_diff.solve(s.reshape((N*N,))).reshape((N,N))

                q_star = r_star - Q[...,i][n:-n,n:-n]
                Q_star[...,i][n:-n,n:-n] = q_star
        
        grid.applyBCs(Q_star, boundary_conditions=self.boundary_conditions)
            
        #plottingHelper = PlottingHelper()
        #plot = plottingHelper.get_plotter()
        #plot.clf()
        ##plot.plot(Q.view(model.state).temp[Nx/2,:15], label="Q.temp")
        ##plot.plot(Q_star.view(model.state).temp[Nx/2,:15], label="Q_star.temp")
        ##plot.plot(Q.view(model.state).temp[Nx/2,:15] - Q_star.view(model.state).temp[Nx/2,:15], label="Q.temp")
        #plot.plot(-dt*conv_Q.view(model.state).temp[Nx/2,:15], label="Q.temp")
        #plot.legend()
        #plot.draw()
        #raw_input()

        #print_grid(Q_star.view(model.state).rho)
        #if debug:
            #plotting_routine(Q=Q_star, grid=grid, model=model, test=test, n_steps=n_steps, t=t, num_scheme="BCG")
            #raw_input()
            
            
        #local_bcs = { 
                #0: (BCs.SolidWall(), BCs.SolidWall(), BCs.ZeroGradient(), BCs.ZeroGradient()),
                #1: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.SolidWall(), BCs.SolidWall()),
                #2: (BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient(), BCs.ZeroGradient())}
        #grid.applyBCs(Q_star, boundary_conditions=local_bcs)

        # 3.    Solve projection on U* to calculate U_n+1

        PROJECTION_TYPE = 0
        # 0: project U*
        # 1: project V=(U* - Un)/dt

        if PROJECTION_TYPE == 0:
            Q_p = 0.5*(Q_star + Q)
            Q_p = Q_p.view(model.state)
        

            (a_x, a_y) = (
                            model.getLaplacianCoefficients(Q_p, axis=0, grid=grid, boundary_conditions=self.boundary_conditions)/(dx*dx),
                            model.getLaplacianCoefficients(Q_p, axis=1, grid=grid, boundary_conditions=self.boundary_conditions)/(dy*dy),
                            )
            a = (a_x, a_y)
            #(a_x, a_y) = (
                    #1.0/( 0.5*(Q_p.rho[1:2,n:-n]+Q_p.rho[2:-1,n:-n]) )/(dx*dx),
                    #1.0/( 0.5*(Q_p.rho[n:-n,1:-2]+Q_p.rho[n:-n,2:-1]) )/(dy*dy),
                            #)
            #a = (a_x, a_y)
        
            D_p = discrete_nonlinear_laplacian_2d(N, a, self.boundary_conditions_pressure)
            # initiate multigrid solver
            ml_p = pyamg.ruge_stuben_solver(D_p)

            # 3.a       divergence of U*, using central differences (grids will be disjoint, but this is what GCB 1999 does, so we'll try)
            (x_, y_)  = grid.getCellCenterPositions(n=num_ghost_cells)
            (c_x, c_y) = (
                            model.getDivergenceCoefficients(y_),
                            model.getDivergenceCoefficients(y_),
                        )
            (u_star, v_star) = (Q_star.view(model.state).x_velocity, Q_star.view(model.state).y_velocity)
            (cu_star, cv_star) = (u_star*c_x, v_star*c_y)
            #r = 0.5/dt*( 
                    #(u_star[4:,2:-2]-u_star[:-4,2:-2])/(2.0*dx) 
                    #+ (v_star[2:-2,4:] - v_star[2:-2,:-4])/(2.0*dy))
            r = 0.5/dt*( 
                    (cu_star[3:-1,2:-2] - cu_star[1:-3,2:-2])/(dx) 
                    + (cv_star[2:-2,3:-1] - cv_star[2:-2,1:-3])/(dy))
            
            # 3.b       solve!
            #(phi_new, info) = scipy.sparse.linalg.cg(D_p, r.ravel(), tol=1.0e-20)
            phi = ml_p.solve(r.ravel(), tol=self.tol)
            phi = phi.reshape(N)
            
            # 3.c       calculate gradient of pseudo-pressure field
            #           the projection step only gives us the pseudo-pressure at cell centers of cells inside the domain
            #           which however is not enough to calculate the gradient of phi on the cells near the boundary (we
            #           would need the first row of ghost cells). To remedy this we approximate the gradient of phi to be linear
            #           near the boundary and then extrapolate the gradient.
            #           (my newest attempt is to extrapolate the velocity correction instead).

            dphi_dx = np.empty((Nx-2,Ny))
            dphi_dx = (phi[2:,:] - phi[:-2,:])/(2.0*dx)

            dphi_dy = np.empty((Nx,Ny-2))
            dphi_dy = (phi[:,2:] - phi[:,:-2])/(2.0*dy)

            #import plotting
            #plottingHelper = plotting.PlottingHelper()
            #plot = plottingHelper.getPlotter()
            #plot.clf()
            ##plot.plot(dphi_dy[Nx/2,:6], label='dpdy', marker='x')
            ##plot.plot(Q.view(model.state).y_velocity[Nx/2,:], label='v', marker='x')
            ##dv = model.getPressureCoefficients(Q=Q_p, grid=grid, boundary_conditions=test.boundary_conditions)
            #dv = model.getPressureCoefficients(Q=Q_p, grid=grid, boundary_conditions=test.boundary_conditions)
            #plot.plot((dphi_dy*dv)[Nx/2,:4], label='dv', marker='x')
            #plot.legend()
            #plot.plot()
            #plot.draw()
            #raw_input()

            kk = model.getPressureCoefficients(Q=Q_p, grid=grid, boundary_conditions=self.boundary_conditions)[Nx/2,:]
            dphi_dy[:,0] #*= kk[1]/kk[0]

            
            #plottingHelper = PlottingHelper()
            #plot = plottingHelper.get_plotter()
            #plot.clf()
            #(x, y) = grid.getCellCenterPositions()
            #plot.plot(y[Nx/2,:], phi_new[Nx/2,:], marker="x", label="phi")
            #plot.plot(y[Nx/2,:], phi_new[0,:], marker="x", label="phi edge")
            #plot.twinx()
            #plot.plot(y[Nx/2,:], Q.view(model.state).y_velocity[Nx/2,n:-n], marker="x", label="Q_old.v")
            #plot.plot(y[Nx/2,:], -dt*grad_phi_y[Nx/2]*model.getPressureCoefficients(Q=Q, y=y, grid=grid, boundary_conditions=test.boundary_conditions)[Nx/2], marker="x", label="Q_new.v1")
            #plot.plot(y[Nx/2,:], -dt*grad_phi_y[Nx/2]*model.getPressureCoefficients(Q=Q_star, y=y, grid=grid, boundary_conditions=test.boundary_conditions)[Nx/2], marker="x", label="Q_new.v2")
            #plot.plot(y[Nx/2,:], -dt*grad_phi_y[Nx/2]*model.getPressureCoefficients(Q=0.5*(Q_star+Q), y=y, grid=grid, boundary_conditions=test.boundary_conditions)[Nx/2], marker="x", label="Q_new.v3")
            #plot.legend()
            #plot.grid(True)
            #plot.draw()
            #raw_input()
            
            # 3.d       update the velocity field
            pos = grid.getCellCenterPositions()
            #Q_p = 0.5*(Q_star + Q)
            Q = Q_star
            Q_star = Q_star.view(model.state)

            du = np.zeros((Nx,Ny))
            dv = np.zeros((Nx,Ny))

            du[1:-1] = -dphi_dx*model.getPressureCoefficients(Q=Q_p, grid=grid, boundary_conditions=self.boundary_conditions)[1:-1,:]
            dv[:,1:-1] = -dphi_dy*model.getPressureCoefficients(Q=Q_p, grid=grid, boundary_conditions=self.boundary_conditions)[:,1:-1]
            du[0,:] = 2.0*du[1,:] - du[2,:]
            du[-1,:] = 2.0*du[-2,:] - du[-3,:]
            dv[:,0] = 2.0*dv[:,1] - dv[:,2]
            dv[:,-1] = 2.0*dv[:,-2] - dv[:,-3]

            Q.view(model.state).x_velocity[n:-n,n:-n] += du*dt
            Q.view(model.state).y_velocity[n:-n,n:-n] += dv*dt
            
            #Q.view(model.state).x_velocity[n:-n,n:-n] = Q_star.x_velocity[n:-n,n:-n] - grad_phi_x*dt/Q_p.view(model.state).rho[n:-n,n:-n]
            #Q.view(model.state).y_velocity[n:-n,n:-n] = Q_star.y_velocity[n:-n,n:-n] - grad_phi_y*dt/Q_p.view(model.state).rho[n:-n,n:-n]
            #Q.view(model.state).x_velocity[n:-n,n:-n] += dt*(V[n:-n,n:-n,0] - grad_phi_x*model.getPressureCoefficients(Q=Q_half_t, y=y, grid=grid, boundary_conditions=test.boundary_conditions))
            #Q.view(model.state).y_velocity[n:-n,n:-n] += dt*(V[n:-n,n:-n,1] - grad_phi_y*model.getPressureCoefficients(Q=Q_half_t, y=y, grid=grid, boundary_conditions=test.boundary_conditions))
            

            #plottingHelper = PlottingHelper()
            #plot = plottingHelper.get_plotter()
            #plot.clf()
            #(x, y) = grid.getCellCenterPositions()
            ##plot.plot(Q.view(model.state).y_velocity[Nx/2]/Q.view(model.state).rho[Nx/2])
            ##plot.plot(Q.view(model.state).y_velocity[Nx/2]/Q_star.view(model.state).rho[Nx/2])
            #plot.plot(Q.view(model.state).y_velocity[Nx/2], color='black')
            #plot.twinx()
            #plot.plot(Q.view(model.state).rho[Nx/2], label='Q.rho')
            #plot.plot(Q_star.view(model.state).rho[Nx/2], label='Q-star.rho')
            #plot.legend()
            #plot.draw()
            #raw_input()
            
            for i in range(2, model.state.num_components):
                Q[...,i] = Q_star[...,i]
            grid.applyBCs(Q, self.boundary_conditions)
            
            #grad_p[...,0] += grad_phi_x
            #grad_p[...,1] += grad_phi_y
            #grad_p[...,0] = grad_phi_x
            #grad_p[...,1] = grad_phi_y
            self.grad_p[...,0] = du
            self.grad_p[...,1] = dv
            

        elif PROJECTION_TYPE == 1:
            # NEW:  make projection on V=(U* - Un)/dt instead
            raise Exception("The V* projection doesn't work yet, gradP is not treated correctly")
            V = (Q_star - Q)/dt
            Q_half_t = 0.5*(Q_star + Q) # only used for the density component, need rho^(n+1/)
        
            (a_x, a_y) = (
                            model.getLaplacianCoefficients(Q_half_t, axis=0, grid=grid, boundary_conditions=self.boundary_conditions)/(dx*dx),
                            model.getLaplacianCoefficients(Q_half_t, axis=1, grid=grid, boundary_conditions=self.boundary_conditions)/(dy*dy),
                            )
            a = (a_x, a_y)
        
            D_p = discrete_nonlinear_laplacian_2d(N, a, self.boundary_conditions_pressure)
            # initiate multigrid solver
            ml_p = pyamg.ruge_stuben_solver(D_p)

            #print "plotting V"
            #plotting_routine(V, grid, model, num_ghost_cells, test)
            #print_asymmetry(V, comp=0, sign=1)
            #print_asymmetry(V, comp=1, sign=-1)
            #print_asymmetry(V, comp=2, sign=-1)
            #raw_input()

            r = 1.0*( 
                    (V[...,0][3:-1,2:-2] - V[...,0][1:-3,2:-2])/(dx) 
                    + (V[...,1][2:-2,3:-1] - V[...,1][2:-2,1:-3])/(dy))


            #print_grid(Q_star.view(model.state).x_velocity - 1.0)
            #return 0
            #print_grid(div_u)
            #print "sss"
            #print Q_star.shape
            #print np.max(np.abs(r))

            #print "r"
            #print_asymmetry(r, sign=-1)
            #raw_input()

            #print "r="
            #print_grid(r)
            #import scipy.constants
            #print "%.20f %.20f" % (r[0,0], scipy.constants.g)
            ##raise KeyboardInterrupt
        
            # 3.b       solve!
            #(phi_new, info) = scipy.sparse.linalg.cg(D_p, r.ravel(), tol=1.0e-20)
            phi_new = ml_p.solve(r.ravel(), tol=1.0e-20)
            phi_new = phi_new.reshape(N)
            
            phi = np.zeros((Nx+2,Ny+2))
            phi[1:-1,1:-1] = phi_new
            phi[0,1:-1] = 2.0*phi[1,1:-1] - phi[2,1:-1]
            phi[-1,1:-1] = 2.0*phi[-2,1:-1] - phi[-3,1:-1]
            phi[:,0] = 2.0*phi[:,1] - phi[:,2]
            phi[:,-1] = 2.0*phi[:,-2] - phi[:,-3]
        
            #print "phi"
            #print_asymmetry(phi, sign=-1)
            
            #plottingHelper = PlottingHelper()
            #plot = plottingHelper.get_plotter()
            #fig = plot.figure(5)
            #plot.clf()
            #ax = fig.add_subplot(111, projection='3d')
            #(x, y) = grid.getCellCenterPositions()
            #ax.plot_wireframe(x, y, phi_new)
            #plot.draw()
            #raw_input()

            # 3.c       calculate gradient of pseudo-pressure field
            grad_phi_x = (phi[2:,1:-1] - phi[:-2,1:-1])/(2.0*dx)
            grad_phi_y = (phi[1:-1,2:] - phi[1:-1,:-2])/(2.0*dy)


        
            #(x, y) = grid.getCellCenterPositions()
            #dQQ = dt*V[n:-n,n:-n,1]
            #dQQ2 = dt*grad_phi_y*model.getPressureCoefficients(Q=Q_half_t, grid=grid, y=y, boundary_conditions=test.boundary_conditions)
            
            #plottingHelper = PlottingHelper()
            #plot = plottingHelper.get_plotter()
            #plot.clf()
            #(x, y) = grid.getCellCenterPositions()
            #plot.plot(y[Nx/2,:], phi_new[Nx/2,:], marker="x", label="phi")
            #plot.plot(y[Nx/2,:], phi_new[0,:], marker="x", label="phi edge")
            #plot.twinx()
            #plot.plot(y[Nx/2,:], dQQ2[Nx/2,:], marker="x", label="dQQ2")
            #plot.plot(y[Nx/2,:], dQQ[Nx/2,:], marker="x", label="dQQ")
            #plot.plot(y[Nx/2,:], Q.view(model.state).y_velocity[Nx/2,n:-n], marker="x", label="Q_old.v")
            #plot.plot(y[Nx/2,:], Q.view(model.state).y_velocity[Nx/2,n:-n] + dQQ[Nx/2,:] - dQQ2[Nx/2,:], marker="x", label="Q_new.v")
            #plot.legend()
            #plot.grid(True)
            #plot.draw()
            #raw_input()

            # 3.d       update the velocity field
            #Q.view(model.state).x_velocity[n:-n,n:-n] -= grad_phi_x*dt*model.getPressureCoefficients(Q, grid, boundary_conditions=test.boundary_conditions)
            #Q.view(model.state).y_velocity[n:-n,n:-n] -= grad_phi_y*dt*model.getPressureCoefficients(Q, grid, boundary_conditions=test.boundary_conditions)
            (x, y) = grid.getCellCenterPositions()
            Q.view(model.state).x_velocity[n:-n,n:-n] += dt*(V[n:-n,n:-n,0] - grad_phi_x*model.getPressureCoefficients(Q=Q_half_t, y=y, grid=grid, boundary_conditions=self.boundary_conditions))
            Q.view(model.state).y_velocity[n:-n,n:-n] += dt*(V[n:-n,n:-n,1] - grad_phi_y*model.getPressureCoefficients(Q=Q_half_t, y=y, grid=grid, boundary_conditions=self.boundary_conditions))
            
            for i in range(2, model.state.num_components):
                Q[...,i] = Q_star[...,i]
            grid.applyBCs(Q, self.boundary_conditions)
            
            #grad_p[...,0] += grad_phi_x
            #grad_p[...,1] += grad_phi_y
        
        """
        # 3.a.i     apply boundary conditions dphi/dn (this is where outflow BCs, will one day be incorporated)
        #b_and_c_on_boundary = model.calculateBuoyancyTermOnBoundary(Q=Q, grid=grid)
        #for i in range(4):
            ## Neither a nor r include ghost cells, therefore to index the outer most rows we have num_ghost_cells=0
            #edge_slice = grid.sliceFromEdge(edge_i=i, row_num=0)
            #dx_ = grid.getGridSpacing()[i/2]
            #print i, r[edge_slice].shape, b_and_c_on_boundary[i].shape
            #r[edge_slice] += 1.0/dx_ * b_and_c_on_boundary[i]
        """



def print_asymmetry(Q, comp = None, sign = 1):
    if comp is None:
        values = (Q[::-1,...] + sign*Q)
        (Nx, Ny) = Q.shape
    else:
        values = (Q[::-1,...] + sign*Q)[...,comp]
        (Nx, Ny, c) = Q.shape
    if Nx < 12:
        print "asymmetry"
        print_grid(values)
    print "max asymmetry=", np.max(np.abs(values))

def calc_slopes(Q, dx, axis):
    """
        Fourth-order monotonicity limited slope approximation, as by Bell, Colella & Howell 1991

        Calculation is done on first component of Q
    """

    #print "## FUU", axis
    #print_grid(Q[...,1])


    (q_l, q_c, q_r) = (Q[aslice(axis,None,-2)], Q[aslice(axis,1,-1)], Q[aslice(axis,2,None)])
    # (q_i+1,j - q_i,j) * (q_i,j - q_i-1,j)
    d_lim_q_cond = (q_r - q_c)*(q_c - q_l) > 0

    d_lim_q = np.zeros(d_lim_q_cond.shape)
    d_lim_q[d_lim_q_cond] = np.minimum( 2.0*np.abs(q_r-q_c)[d_lim_q_cond], 2.0*np.abs(q_c-q_l)[d_lim_q_cond] )

    #print "d_lim_q"
    #print_asymmetry(d_lim_q[...,1], sign=-1)
    #print_grid(d_lim_q[...,1])

    d_p_q = np.copysign( np.minimum( 2.0*np.abs(q_r-q_l), d_lim_q ), np.sign(q_r-q_l) )

    #print "d_p_q"
    #print_asymmetry(d_p_q[...,1], sign=1)
    #print_grid(d_p_q[...,1])

    # q_l and q_r must be reduced in size since this is a five-point stencil
    (q_l, q_r) = (q_l[aslice(axis,1,-1)], q_r[aslice(axis,1,-1)])
    (d_p_q_l, d_p_q_c, d_p_q_r) = (d_p_q[aslice(axis,None,-2)], d_p_q[aslice(axis,1,-1)], d_p_q[aslice(axis,2,None)])
    
    #print "q_l"
    #print_grid(q_l[...,1])
    #print "q_r"
    #print_grid(q_r[...,1])
    #DDD = np.abs( 2.0*(q_r-q_l)/3.0 - (d_p_q_r+d_p_q_l)/6.0 )
    #DD = d_p_q_c
    #print "DDD"
    #print_grid(DDD[...,1])
    #print_grid(DD[...,1])
    
    d_q = np.copysign( np.minimum( np.abs( 2.0*(q_r-q_l)/3.0 - (d_p_q_r+d_p_q_l)/6.0 ), d_p_q_c), (q_r-q_l) )

    #print "d_q"
    #print_grid(d_q[...,1])
    #raw_input()

    return d_p_q[aslice(axis, 1, -1)]/dx
    #return d_q/dx

def calc_hat_states(Q, dx, dt, model, axis, grid):
    #print "## calc_hat_states"
    #print "Q.x_velocity"
    #print_grid(Q.view(model.state).x_velocity)
    #print_asymmetry(Q.view(model.state).x_velocity, sign=1)
    #print "Q.y_velocity"
    #print_grid(Q.view(model.state).y_velocity)
    #print_asymmetry(Q.view(model.state).y_velocity, sign=-1)

    Q = Q.view(model.state)
    (Nx, Ny) = grid.getNumCells()
    p_axis = (axis + 1) % 2
    dQdx = calc_slopes(Q, dx, axis=axis)[aslice(p_axis,2,-2)]
    #print "axis=", axis, "p_axis=", p_axis

    #print "dQdx.y_velocity (all)"
    #print_grid(dQdx.view(model.state).y_velocity)
    #raw_input()

    # extrapolate left states
    # center from which we extrapolate changes for every interface
    Q_c = Q[aslice(axis,2,-3)][aslice(p_axis,2,-2)]
    u_c = Q_c[...,axis]
    
    #print "Q_c.y_velocity"
    #print_grid(Q_c.view(model.state).y_velocity)
    dQdx_c = dQdx[aslice(axis,None,-1)]
    #print "dQdx.y_velocity"
    #print_grid(dQdx_c.view(model.state).y_velocity)
    #print "#### HERE: ", dQdx_c.shape
    
    #s_l = np.ones((Nx-1+axis, Ny-axis, Q_c.num_components))
    #s_l_i = u_c >= 0.0
    #s_l[s_l_i] = 1.0
    s_l = 1.0

    u_c_n = stack_multiple_copies(u_c, Q_c.num_components)

    Q_h_l = Q_c + s_l*(dx/2.0 - dt/2.0*u_c_n)*dQdx_c

    # extrapolate right states
    Q_c = Q[aslice(axis,3,-2)][aslice(p_axis,2,-2)]
    u_c = Q_c[...,axis]
    
    #print "Q_c.y_velocity"
    #print_grid(Q_c.view(model.state).y_velocity)
    dQdx_c = dQdx[aslice(axis,1,None)]
    #print "dQdx.y_velocity"
    #print_grid(dQdx_c.view(model.state).y_velocity)
    #print "#### HERE: ", dQdx_c.shape

    #s_r = np.ones((Nx-1+axis, Ny-axis, Q_c.num_components))
    #s_r_i = u_c <= 0.0
    #s_r[s_r_i] = 1.0
    s_r = 1.0

    u_c_n = stack_multiple_copies(u_c, Q_c.num_components)

    Q_h_r = Q_c - s_r*(dx/2.0 + dt/2.0*u_c_n)*dQdx_c
    
    #print "Q_h_l.y_velocity"
    #print_grid(Q_h_l.view(model.state).y_velocity)
    #print "Q_h_r.y_velocity"
    #print_grid(Q_h_r.view(model.state).y_velocity)

    #raw_input()

    return (Q_h_l, Q_h_r)

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

def calc_RP(Q_h_l, Q_h_r, model, axis):
    #print "Q_h_l.x_velocity"
    #print_grid(Q_h_l.view(model.state).x_velocity)
    #print "Q_h_r.x_velocity"
    #print_grid(Q_h_r.view(model.state).x_velocity)
    
    #print "Q_h_l.y_velocity"
    #print_grid(Q_h_l.view(model.state).y_velocity)
    #print "Q_h_r.y_velocity"
    #print_grid(Q_h_r.view(model.state).y_velocity)

    #raw_input()

    # x-direction
    # first solve for the x_velocity
    u_h_l = Q_h_l[...,axis]
    u_h_r = Q_h_r[...,axis]
    # find indecies for cells in which the left velocity should be used
    #u_h_int_x_l_i = (u_h_l > 0.0) * (u_h_l + u_h_r > 0.0)
    u_h_int_x_l_i = (u_h_l > epsilon) * (u_h_l + u_h_r > epsilon)
    # and for cells for which the right velocity should be used
    #u_h_int_x_r_i = (u_h_r < 0.0) * (u_h_l + u_h_r < 0.0)
    u_h_int_x_r_i = (u_h_r < -epsilon) * (u_h_l + u_h_r < -epsilon)

    # fix for waves meeting at interface, it seems the the conditionals in BCG are not
    # mutually exclusive
    u_h_int_x_c_i = (u_h_r <= 0.0) * (u_h_l >= 0.0)

    #print "velocities"
    #print_grid(u_h_l)
    #print_grid(u_h_r)
    #print "indecies"
    #print_grid(u_h_int_x_l_i)
    #print_grid(u_h_int_x_r_i)
    #raw_input()
    # set the interfaces velocities with the indecies found
    u_h_int_x = np.zeros(u_h_l.shape)
    u_h_int_x[u_h_int_x_l_i] = u_h_l[u_h_int_x_l_i]
    u_h_int_x[u_h_int_x_r_i] = u_h_r[u_h_int_x_r_i]
    u_h_int_x[u_h_int_x_c_i] = 0.0

    Q_h_int_x = upwind_states(Q_h_l, Q_h_r, u_h_int_x)
    Q_h_int_x[...,axis] = u_h_int_x  # Set the velocity using the velocities from the RP above (this may not be necessary)

    return Q_h_int_x

def stack_multiple_copies(v, copies=3):
    (nx, ny) = v.shape
    Q = np.empty((nx, ny, copies))
    for i in range(copies):
        Q[...,i] = v
    return Q


def calc_fivepoint_laplacian(Q, dx, dy):

    #print "####"
    #print "Q.x_velocity"
    #print_asymmetry(Q[...,0], sign=1)
    #print "Q.y_velocity"
    #print_asymmetry(Q[...,1], sign=-1)
    #raw_input()

    return (Q[:-2,1:-1,...] - 2.0*Q[1:-1,1:-1,...] + Q[2:,1:-1,...])/(dx*dx) + (Q[1:-1,:-2,...] - 2.0*Q[1:-1,1:-1,...] + Q[1:-1,2:,...])/(dy*dy)

