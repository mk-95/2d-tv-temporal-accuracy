import numpy as np
from functions import func
import time
import singleton_classes as sc
import statistics
import matplotlib.pyplot as plt

def error_tv_time_dependent_bcs_FE (steps = 3,return_stability=False, name='', guess=None, project=[],alpha=0.99):
    probDescription = sc.ProbDescription()
    f = func(probDescription)
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()

    a = 2 * np.pi
    b = 2 * np.pi
    uexact = lambda a, b, x, y, t: 1 - np.cos(a * (x - t)) * np.sin(b * (y - t)) * np.exp(-(a ** 2 + b ** 2) * μ * t)
    vexact = lambda a, b, x, y, t: 1 + np.sin(a * (x - t)) * np.cos(b * (y - t)) * np.exp(-(a ** 2 + b ** 2) * μ * t)
    pexact = lambda x, y, t: (-8*np.sin(np.pi*t)**4*np.sin(np.pi*y)**4 - 2*np.sin(np.pi*t)**4 - 2*np.sin(np.pi*y)**4 - 5*np.cos(2*np.pi*t)/2 + 5*np.cos(4*np.pi*t)/8 - 5*np.cos(2*np.pi*y)/2 + 5*np.cos(4*np.pi*y)/8 - np.cos(np.pi*(2*t - 4*y))/4 + np.cos(np.pi*(2*t - 2*y)) + np.cos(np.pi*(2*t + 2*y)) - np.cos(np.pi*(2*t + 4*y))/4 - 3*np.cos(np.pi*(4*t - 4*y))/16 - np.cos(np.pi*(4*t - 2*y))/4 - np.cos(np.pi*(4*t + 2*y))/4 + np.cos(np.pi*(4*t + 4*y))/16 + 27/8)*np.exp(-16*np.pi**2*μ*t) - np.exp(-16*np.pi**2*μ*t)*np.cos(np.pi*(-4*t + 4*x))/4
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
    u0[1:-1, 1:] = uexact(a, b, xu, yu, t)  # initialize the interior of u0
    # same thing for the y-velocity component
    v0 = np.zeros([ny +2, nx+2]) # include ghost cells
    v0[1:, 1:-1] = vexact(a, b, xv, yv, t)

    print('div_before_bcs= {}'.format(np.linalg.norm(f.div(u0,v0))))
    # print('right wall :, yv= {}'.format(yv[:,-1]))

    u_bc_top_wall = lambda t:lambda xv: uexact(a,b,xu[-1,:],np.ones_like(xu[-1,:]),t)
    u_bc_bottom_wall = lambda t:lambda xv: uexact(a,b,xu[0,:],np.zeros_like(xu[0,:]),t)
    u_bc_right_wall = lambda t:lambda yv: uexact(a,b,xu[:,-1],yu[:,-1],t)
    u_bc_left_wall = lambda t:lambda yv: uexact(a,b,xu[:,0],yu[:,0],t)

    v_bc_top_wall = lambda t:lambda xv: vexact(a,b,xv[-1,:],yv[-1,:],t)
    v_bc_bottom_wall = lambda t:lambda xv: vexact(a,b,xv[0,:],yv[0,:],t)
    v_bc_right_wall = lambda t:lambda yv: vexact(a,b,np.ones_like(yv[:,-1]),yv[:,-1],t)
    v_bc_left_wall = lambda t:lambda yv: vexact(a,b,np.zeros_like(yv[:,0]),yv[:,0],t)

    # plt.imshow(u0,origin='bottom')
    # plt.show()
    # pressure
    def pressure_bcs(p,t):
        # # right wall
        # p[1:-1,-1] = 2*pexact(np.ones_like(ycc[:,-1]),ycc[:,-1],t)/ np.sum(pexact(np.ones_like(ycc[:,-2]),ycc[:,-2],t).ravel()) *np.sum(p[1:-1,-2].ravel())  -p[1:-1,-2]
        # # left wall
        # p[1:-1,0] = 2*pexact(np.zeros_like(ycc[:,0]),ycc[:,0],t)/np.sum(pexact(np.zeros_like(ycc[:,1]),ycc[:,1],t).ravel())*np.sum(p[1:-1,1].ravel()) -p[1:-1,1]
        # # top wall
        # p[-1,1:-1] = 2*pexact(np.ones_like(ycc[-1,:]),ycc[-1,:],t)/np.sum(pexact(np.ones_like(ycc[-2,:]),ycc[-2,:],t).ravel())*np.sum(p[-2,1:-1].ravel()) - p[-2,1:-1]
        # # bottom wall
        # p[0, 1:-1] = 2 * pexact(np.zeros_like(ycc[0,:]),ycc[0,:],t)/np.sum(pexact(np.zeros_like(ycc[1,:]),ycc[1,:],t).ravel())*np.sum(p[1, 1:-1].ravel()) - p[1, 1:-1]
        #
        # try extrapolation
        # right wall
        p[1:-1, -1] = (p[1:-1, -2] -p[1:-1, -3]) + p[1:-1, -2]
        # left wall
        p[1:-1, 0] = -(p[1:-1, 2] -p[1:-1, 1]) + p[1:-1, 1]
        # top wall
        p[-1, 1:-1] = (p[-2, 1:-1] - p[-3, 1:-1]) + p[-2, 1:-1]
        # bottom wall
        p[0, 1:-1] = -(p[2, 1:-1] - p[1, 1:-1]) + p[1, 1:-1]
    p_bcs = lambda t:lambda p:pressure_bcs(p,t)
    # apply bcs
    f.top_wall(u0,v0,u_bc_top_wall(t+dt),v_bc_top_wall(t+dt))
    f.bottom_wall(u0,v0, u_bc_bottom_wall(t+dt),   v_bc_bottom_wall(t+dt))
    f.right_wall(u0,v0,u_bc_right_wall(t+dt),v_bc_right_wall(t+dt))
    f.left_wall(u0,v0,u_bc_left_wall(t+dt),v_bc_left_wall(t+dt))

    print('div_after_bcs= {}'.format(np.linalg.norm(f.div(u0, v0))))

    # plt.imshow(u0, origin='bottom')
    # plt.show()

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
    Coef = f.A_Lid_driven_cavity()

    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        # rk coefficients
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 =  np.zeros_like(u)
        time_start = time.clock()
        print('div_u= {}'.format(np.linalg.norm(f.div(u, v))))

        time_start = time.clock()

        # Au1
        urhs1 = f.urhs_bcs(u, v)
        vrhs1 = f.vrhs_bcs(u, v)

        # divergence of u1
        div_n = np.linalg.norm(f.div(u, v).ravel())
        print('        divergence of u1 = ', div_n)
        ## stage 2
        print('    Stage 2:')
        print('    --------')
        uh = u +  dt * (urhs1)
        vh = v +  dt * (vrhs1)

        f.top_wall(uh, vh, u_bc_top_wall(t + dt), v_bc_top_wall(t + dt))
        f.bottom_wall(uh, vh, u_bc_bottom_wall(t + dt), v_bc_bottom_wall(t + dt))
        f.right_wall(uh, vh, u_bc_right_wall(t + dt), v_bc_right_wall(t + dt))
        f.left_wall(uh, vh, u_bc_left_wall(t + dt), v_bc_left_wall(t + dt))

        unp1, vnp1, press, iter1 = f.ImQ_bcs(uh, vh, Coef, pn, p_bcs((t + dt)))

        # apply bcs
        f.top_wall(unp1, vnp1, u_bc_top_wall(t + dt), v_bc_top_wall(t + dt))
        f.bottom_wall(unp1, vnp1, u_bc_bottom_wall(t + dt), v_bc_bottom_wall(t + dt))
        f.right_wall(unp1, vnp1, u_bc_right_wall(t + dt), v_bc_right_wall(t + dt))
        f.left_wall(unp1, vnp1, u_bc_left_wall(t + dt), v_bc_left_wall(t + dt))

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
        # if count % 1 ==0:
        #     # plt.imshow(unp1[1:-1,1:]-uexact(a,b,xu,yu,t),origin='bottom')
        #     plt.imshow(unp1[1:-1,1:],origin='bottom')
        #     plt.colorbar()
        #     # divu = f.div(unp1,vnp1)
        #     # plt.imshow(divu[1:-1,1:-1], origin='bottom')
        #     # plt.colorbar()
        #     # ucc = 0.5 * (u[1:-1, 2:] + u[1:-1, 1:-1])
        #     # vcc = 0.5 * (v[2:, 1:-1] + v[1:-1, 1:-1])
        #     # speed = np.sqrt(ucc * ucc + vcc * vcc)
        #     # uexact = 4 * 1.5 * ycc * (1 - ycc)
        #     # plt.plot(uexact, ycc, '-k', label='exact')
        #     # plt.plot(ucc[:, int(8 / dx)], ycc, '--', label='x = {}'.format(8))
        #     # plt.contourf(xcc, ycc, speed)
        #     # plt.colorbar()
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
# probDescription = ProbDescription(N=[32,32],L=[1,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# print('dt = ',dt)
# probDescription.set_dt(0.001)
# error_tv_time_dependent_bcs_FE (steps = 2000)