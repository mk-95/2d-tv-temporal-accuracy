import numpy as np
from functions import func
import time
import singleton_classes as sc
import statistics
import matplotlib.pyplot as plt

def error_channel_flow_FE (steps = 3,return_stability=False, name='', guess=None, project=[],alpha=0.99):
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
    u0 = np.zeros([ny+2, nx + 2]) # include ghost cells
    # same thing for the y-velocity component
    v0 = np.zeros([ny +2, nx+2]) # include ghost cells

    u_bc_top_wall = lambda xv: 0
    u_bc_bottom_wall = lambda xv: 0
    u_bc_right_wall = lambda u:lambda yv: u
    u_bc_left_wall = lambda yv: 1

    v_bc_top_wall = lambda xv: 0
    v_bc_bottom_wall = lambda xv: 0
    v_bc_right_wall = lambda yv: 0
    v_bc_left_wall = lambda yv: 0

    # pressure
    def pressure_right_wall(p):
        # pressure on the right wall
        p[1:-1, -1] = -p[1:-1, -2]

    p_bcs = lambda p: pressure_right_wall(p)
    # apply bcs
    f.top_wall(u0,v0,u_bc_top_wall,v_bc_top_wall)
    f.bottom_wall(u0,v0, u_bc_bottom_wall,   v_bc_bottom_wall)
    f.right_wall(u0,v0,u_bc_right_wall(u0[1:-1, -1]),v_bc_right_wall)
    f.left_wall(u0,v0,u_bc_left_wall,v_bc_left_wall)


    # initialize the pressure
    p0 = np.zeros([nx+2,ny+2]); # include ghost cells

    #declare unp1
    unp1 = np.zeros_like(u0)
    vnp1 = np.zeros_like(v0)

    div_np1= np.zeros_like(p0)
    # a bunch of lists for animation purposes
    usol=[]
    usol.append(u0)

    vsol=[]
    vsol.append(v0)

    psol = []
    psol.append(p0)
    iterations = [0]
    Coef = f.A_channel_flow()
    total_iters = 0
    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        # rk coefficients
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 =  np.zeros_like(u)
        time_start = time.clock()
        uhnp1 = u + dt*f.urhs_bcs(u, v)
        vhnp1 = v + dt*f.vrhs_bcs(u,v)

        f.left_wall(uhnp1, vhnp1, u_bc_left_wall, v_bc_left_wall)

        unp1,vnp1,press, iter = f.ImQ_bcs(uhnp1,vhnp1,Coef,pn,p_bcs)

        total_iters+=iter

        # apply bcs
        f.top_wall(unp1, vnp1, u_bc_top_wall, v_bc_top_wall)
        f.bottom_wall(unp1, vnp1, u_bc_bottom_wall, v_bc_bottom_wall)
        f.right_wall(unp1, vnp1, u_bc_right_wall(unp1[1:-1, -1]), v_bc_right_wall)
        f.left_wall(unp1, vnp1, u_bc_left_wall, v_bc_left_wall)

        time_end = time.clock()
        psol.append(press)
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

        t += dt

        # plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        # if count % 10 ==0:
        #     divu = f.div(unp1,vnp1)
        #     plt.imshow(divu[1:-1,1:-1], origin='bottom')
        #     plt.colorbar()
        #     # ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        #     # vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])
        #     # speed = np.sqrt(ucc * ucc + vcc * vcc)
        #     # uexact = 4 * 1.5 * ycc * (1 - ycc)
        #     # plt.plot(uexact, ycc, '-k', label='exact')
        #     # plt.plot(ucc[:, int(8 / dx)], ycc, '--', label='x = {}'.format(8))
        #     # plt.contourf(xcc, ycc, speed)
        #     # plt.colorbar()
        #     # plt.streamplot(xcc, ycc, ucc, vcc, color='black', density=0.75, linewidth=1.5)
        #     # plt.contourf(xcc, ycc, speed)
        #     # plt.colorbar()
        #     plt.show()
        count += 1

    if return_stability:
        return True
    else:
        return True, [total_iters], True, unp1[1:-1, 1:-1].ravel()

# from singleton_classes import ProbDescription
# #
# Uinlet = 1
# ν = 0.01
# probDescription = ProbDescription(N=[4*32,32],L=[10,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt)
# error_channel_flow_FE (steps = 2000)