import numpy as np
from functions import *
import time
import statistics

def error_FE (dt,μ,n = [32,32],steps = 3,return_iter=False):
    # define exact solutions
    a = 2*np.pi
    b = 2*np.pi
    uexact = lambda a, b, x, y, t: 1 - np.cos(a*(x - t))*np.sin(b*(y - t))*np.exp(-(a**2 + b**2)*μ*t)
    vexact = lambda a, b, x, y, t: 1 + np.sin(a*(x - t))*np.cos(b*(y - t))*np.exp(-(a**2 + b**2)*μ*t)

    #     # define some boiler plate
    nx = n[0]
    ny = n[1]
    # dt =0.09/p # timestep size
    lx = 1.0
    ly = 1.0
    dx = lx/nx
    dy = ly/ny

    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)
    # cell centered coordinates
    xx = np.linspace(dx/2.0,lx - dx/2.0,nx, endpoint=True)
    yy = np.linspace(dy/2.0,ly - dy/2.0,ny, endpoint=True)
    xcc, ycc = np.meshgrid(xx,yy)

    # x-staggered coordinates
    xxs = np.linspace(0,lx,nx+1, endpoint=True)
    xu,yu = np.meshgrid(xxs, yy)

    # y-staggered coordinates
    yys = np.linspace(0,ly,ny+1, endpoint=True)
    xv,yv = np.meshgrid(xx, yys)

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.zeros([ny+2, nx + 2]) # include ghost cells
    u0[1:-1,1:] = uexact(a,b,xu,yu,0) # initialize the interior of u0
    # same thing for the y-velocity component
    v0 = np.zeros([ny +2, nx+2]) # include ghost cells
    v0[1:,1:-1] = vexact(a,b,xv,yv,0)
    periodic_u(u0)
    periodic_v(v0)

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
    Coef = A(nx,ny,dx,dy)

    is_stable =True
    stability_counter =0
    # # u and v num cell centered
    ucc = 0.5*(u0[1:-1,2:] + u0[1:-1,1:-1])
    vcc = 0.5*(v0[2:,1:-1] + v0[1:-1,1:-1])

    uexc = uexact(a,b,xu,yu,t)
    vexc = vexact(a,b,xv,yv,t)
    # u and v exact cell centered
    uexc_cc = 0.5*(uexc[:,:-1] + uexc[:,1:])
    vexc_cc = 0.5*(vexc[:-1,:] + vexc[1:,:])

    # compute of kinetic energy
    ken_new = np.sum(ucc.ravel()**2 +vcc.ravel()**2)/2
    ken_exact = np.sum(uexc_cc.ravel()**2 +vexc_cc.ravel()**2)/2
    ken_old = ken_new
    final_KE = nx*ny
    alpha = 0.999
    target_ke = ken_exact - alpha*(ken_exact-final_KE)
    print('time = ',t)
    print('ken_new = ',ken_new)
    print('ken_exc = ',ken_exact)
    while count < tend:
        print('timestep:{}'.format(count+1))
        print('-----------')
        # rk coefficients
        u = usol[-1].copy()
        v = vsol[-1].copy()
        pn = np.zeros_like(u)
        pnm1 =  np.zeros_like(u)
        time_start = time.clock()
        uhnp1 = u + dt*urhs(u,v,μ,dx,dy,nx,ny)
        vhnp1 = v + dt*vrhs(u,v,μ,dx,dy,nx,ny)

        unp1,vnp1,press, iter = ImQ(uhnp1,vhnp1,Coef,dx,dy,dt,nx,ny,pn)
        time_end = time.clock()
        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(div(unp1,vnp1,dx,dy,nx,ny).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:',residual)
        print('iterations:',iter)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        iterations.append(iter)
        # # u and v num cell centered
        ucc = 0.5*(u[1:-1,2:] + u[1:-1,1:-1])
        vcc = 0.5*(v[2:,1:-1] + v[1:-1,1:-1])

        uexc = uexact(a,b,xu,yu,t)
        vexc = vexact(a,b,xv,yv,t)
        # u and v exact cell centered
        uexc_cc = 0.5*(uexc[:,:-1] + uexc[:,1:])
        vexc_cc = 0.5*(vexc[:-1,:] + vexc[1:,:])
        t += dt

        # compute of kinetic energy
        ken_new = np.sum(ucc.ravel()**2 +vcc.ravel()**2)/2
        ken_exact = np.sum(uexc_cc.ravel()**2 +vexc_cc.ravel()**2)/2
        print('time = ',t)
        print('ken_new = ',ken_new)
        print('target_ken=', target_ke)
        print('ken_exc = ',ken_exact)
        print('(ken_new - ken_old)/ken_old = ',(ken_new - ken_old)/ken_old)
        if (((ken_new - ken_old)/ken_old) > 0 and count>1) or np.isnan(ken_new):
            is_stable = False
            print('is_stable = ',is_stable)
            if stability_counter >3:
                print('not stable !!!!!!!!')
                break
            else:
                stability_counter+=1
        else:
            is_stable = True
            print('is_stable = ',is_stable)
            if ken_new<target_ke and count > 30:
                break
        ken_old = ken_new.copy()

        #plot of the pressure gradient in order to make sure the solution is correct
        # # plt.contourf(usol[-1][1:-1,1:])
        # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        # plt.colorbar()
        # plt.show()
        count+=1
    diff = np.linalg.norm(uexact(a,b,xu,yu,t).ravel()-unp1[1:-1,1:] .ravel(),np.inf)
    print('        error={}'.format(diff))
    if return_iter:
        return diff, [div_np1], is_stable, int(statistics.mean(iterations)), int(sum(iterations))
    else:
        return diff, [div_np1], is_stable, unp1[1:-1,1:].ravel()