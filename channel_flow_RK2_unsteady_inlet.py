import numpy as np
from functions import func
import time
import singleton_classes as sc
import statistics
import matplotlib.pyplot as plt

def error_channel_flow_RK2_unsteady_inlet (steps = 3,return_stability=False, name='heun', guess=None, project=[],theta=None):
    probDescription = sc.ProbDescription()
    f = func(probDescription)
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()

    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)
    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    np.random.seed(123)
    u0 = np.random.rand(ny + 2, nx + 2)/10000  # include ghost cells
    # u0 = np.ones([ny +2, nx+2])# include ghost cells
    # same thing for the y-velocity component
    v0 = np.random.rand(ny + 2, nx + 2)/10000  # include ghost cells
    # v0 = np.ones([ny +2, nx+2])  # include ghost cells

    at = lambda t: (np.pi / 6) * np.sin(t / 2)

    u_bc_top_wall = lambda xv: 0
    u_bc_bottom_wall = lambda xv: 0
    u_bc_right_wall = lambda u: lambda yv: u
    u_bc_left_wall = lambda t: lambda yv: np.cos(at(t))

    v_bc_top_wall = lambda xv: 0
    v_bc_bottom_wall = lambda xv: 0
    v_bc_right_wall = lambda yv: 0
    v_bc_left_wall = lambda t: lambda yv: np.sin(at(t))
    # pressure
    def pressure_right_wall(p):
        # pressure on the right wall
        p[1:-1, -1] = -p[1:-1, -2]

    p_bcs = lambda p: pressure_right_wall(p)
    # apply bcs
    f.top_wall(u0, v0, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0, v0, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0, v0, u_bc_right_wall(u0[1:-1, -1]), v_bc_right_wall)
    f.left_wall(u0, v0, u_bc_left_wall(t), v_bc_left_wall(t))

    Coef = f.A_channel_flow()

    u0_free, v0_free, _, _ = f.ImQ_bcs(u0, v0, Coef, 0, p_bcs)

    f.top_wall(u0_free, v0_free, u_bc_top_wall, v_bc_top_wall)
    f.bottom_wall(u0_free, v0_free, u_bc_bottom_wall, v_bc_bottom_wall)
    f.right_wall(u0_free, v0_free, u_bc_right_wall(u0_free[1:-1, -1]), v_bc_right_wall)
    f.left_wall(u0_free, v0_free, u_bc_left_wall(t), v_bc_left_wall(t))

    print('div_u0=', np.linalg.norm(f.div(u0_free, v0_free).ravel()))

    # initialize the pressure
    p0 = np.zeros([nx+2,ny+2]); # include ghost cells

    #declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1= np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol=[]
    # usol.append(u0)
    usol.append(u0_free)

    vsol=[]
    # vsol.append(v0)
    vsol.append(v0_free)

    psol = []
    psol.append(p0)
    iterations = [0]

    while count < tend:
        print('timestep:{}'.format(count + 1))
        print('-----------')
        # rk coefficients
        RK2 = sc.RK2(name,theta)
        a21 = RK2.a21
        b1 = RK2.b1
        b2 = RK2.b2
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 = np.zeros_like(u)
        # pnm2 = np.zeros_like(u) # only needed for high accurate pressure
        if count > 1: # change the count for 2 if high accurate pressure at time np1 is needed
            pn = psol[-1].copy()
            pnm1 = psol[-2].copy()
            # pnm2 = psol[-3].copy() # only needed for high accurate pressure
            f1x, f1y = f.Guess([pn, pnm1], order=guess, integ='RK2', type=name,theta=theta)
            d2, = project

        elif count <= 1:   # compute pressures for 2 time steps # change the count for 2 if high accurate pressure at time np1 is needed
            d2 = 1
            f1x, f1y = f.Guess([pn, pnm1], order=None, integ='RK2', type=name,theta=theta)

        ## stage 1

        print('    Stage 1:')
        print('    --------')
        time_start = time.clock()
        u1 = u.copy()
        v1 = v.copy()

        # Au1

        # apply boundary conditions before the computation of the rhs
        f.top_wall(u1, v1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u1, v1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u1, v1, u_bc_right_wall(u1[1:-1, -1]), v_bc_right_wall)  # this won't change anything for u2
        f.left_wall(u1, v1, u_bc_left_wall(t), v_bc_left_wall(t))

        urhs1 = f.urhs_bcs(u1, v1)
        vrhs1 = f.vrhs_bcs(u1, v1)

        # divergence of u1
        div_n = np.linalg.norm(f.div(u1, v1).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh2 = u + a21 * dt * (urhs1 - f1x)
        vh2 = v + a21 * dt * (vrhs1 - f1y)

        f.top_wall(uh2, vh2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uh2, vh2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uh2, vh2, u_bc_right_wall(uh2[1:-1, -2]), v_bc_right_wall)  # this won't change anything for u2
        f.left_wall(uh2, vh2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))

        if d2 == 1:
            print('        pressure projection stage{} = True'.format(2))
            u2, v2, _, iter1 = f.ImQ_bcs(uh2, vh2, Coef, pn,p_bcs)
            print('        iterations stage 2 = ', iter1)
        elif d2 == 0:
            u2 = uh2
            v2 = vh2

        # apply bcs
        f.top_wall(u2, v2, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(u2, v2, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(u2, v2, u_bc_right_wall(u2[1:-1, -1]), v_bc_right_wall) # this won't change anything for u2
        f.left_wall(u2, v2, u_bc_left_wall(t+a21*dt), v_bc_left_wall(t+a21*dt))


        div2 = np.linalg.norm(f.div(u2, v2).ravel())
        print('        divergence of u2 = ', div2)
        urhs2 = f.urhs_bcs(u2, v2)
        vrhs2 = f.vrhs_bcs(u2, v2)

        uhnp1 = u + dt * b1 * (urhs1) + dt * b2 * (urhs2)
        vhnp1 = v + dt * b1 * (vrhs1) + dt * b2 * (vrhs2)

        f.top_wall(uhnp1, vhnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(uhnp1, vhnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(uhnp1, vhnp1, u_bc_right_wall(uhnp1[1:-1, -2]), v_bc_right_wall)  # this won't change anything for u2
        f.left_wall(uhnp1, vhnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        unp1, vnp1, press, iter2 = f.ImQ_bcs(uhnp1, vhnp1, Coef, pn,p_bcs)

        # apply bcs
        f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]), v_bc_right_wall) # this won't change anything for unp1
        f.left_wall(unp1, vnp1, u_bc_left_wall(t+dt), v_bc_left_wall(t+dt))

        time_end = time.clock()
        psol.append(press)

        # new_press = 4 * pn - 9 * pnm1 / 2 + 3 * pnm2 / 2  # second order (working)

        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1,vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:',residual)
        print('iterations:',iter)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        iterations.append(iter)

        Min = np.sum(unp1[1:ny+1,1])
        Mout = np.sum(unp1[1:ny+1,nx+1])

        print("Min=",Min)
        print("Mout=",Mout)


        t += dt

        # plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        # if count % 10 ==0:
        #     # divu = f.div(u0_free,v0_free)
        #     # plt.imshow(divu[1:-1,1:-1], origin='bottom')
        #     # plt.colorbar()
        #     ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        #     vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])
        #     speed = np.sqrt(ucc * ucc + vcc * vcc)
        #     # uexact = 4 * 1.5 * ycc * (1 - ycc)
        #     # plt.plot(uexact, ycc, '-k', label='exact')
        #     # plt.plot(ucc[:, int(8 / dx)], ycc, '--', label='x = {}'.format(8))
        #     plt.contourf(xcc, ycc, speed)
        #     plt.colorbar()
        #     # plt.streamplot(xcc, ycc, ucc, vcc, color='black', density=0.75, linewidth=1.5)
        #     # plt.contourf(xcc, ycc, psol[-1][1:-1, 1:-1])
        #     # plt.colorbar()
        #     plt.show()
        count += 1

    if return_stability:
        return True
    else:
        return True, [div_np1], True, unp1[1:-1,1:-1].ravel()


# from singleton_classes import ProbDescription
# #
# Uinlet = 1
# ν = 0.01
# probDescription = ProbDescription(N=[4*32,32],L=[10,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt)
# error_channel_flow_RK2_unsteady_inlet (steps = 100,return_stability=False, name='midpoint', guess=None, project=[1],theta=None)