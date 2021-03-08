import numpy as np
from functions import func
import time
import statistics
import singleton_classes as sc
import scipy
from scipy.optimize import fsolve

def error_DIRK2 (steps=3, return_stability=False,name='midpoint',alpha=0.99,theta=0.25):
    # problem description
    probDescription = sc.ProbDescription()
    f = func(probDescription, 'periodic')
    dt = probDescription.get_dt()
    μ = probDescription.get_mu()
    nx, ny = probDescription.get_gridPoints()
    dx, dy = probDescription.get_differential_elements()
    # define exact solutions
    a = 2 * np.pi
    b = 2 * np.pi
    uf = 1
    vf = 1
    uexact = lambda a, b, x, y, t: uf - np.cos(a * (x - uf * t)) * np.sin(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)
    vexact = lambda a, b, x, y, t: vf + np.sin(a * (x - uf * t)) * np.cos(b * (y - vf * t)) * np.exp(
        -(a ** 2 + b ** 2) * μ * t)

    #     # define some boiler plate
    t = 0.0
    tend = steps
    count = 0
    print('dt=',dt)

    xcc, ycc = probDescription.get_cell_centered()
    xu, yu = probDescription.get_XVol()
    xv, yv = probDescription.get_YVol()

    # initialize velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.
    # Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
    u0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    u0[1:-1, 1:] = uexact(a, b, xu, yu, 0)  # initialize the interior of u0
    # same thing for the y-velocity component
    v0 = np.zeros([ny + 2, nx + 2])  # include ghost cells
    v0[1:, 1:-1] = vexact(a, b, xv, yv, 0)
    f.periodic_u(u0)
    f.periodic_v(v0)

    # initialize the pressure
    p0 = np.zeros([nx + 2, ny + 2]);  # include ghost cells

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

    iterations = []

    Coef = f.A()

    is_stable =True
    stability_counter =0
    total_iteration =0
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
        DIRK2 = sc.DIRK2(name,theta)
        b1 = DIRK2.b1
        b2 = DIRK2.b2
        time_start = time.clock()
        un = usol[-1].copy()
        vn = vsol[-1].copy()
        pn = psol[-1].copy()

        # stage 1:
        #---------

        u1,v1,p1,info = f.DIRK_S1(un,vn,pn,DIRK2)
        print('        number of function calls stage 1: ', info['nfev'])
        rhs_u1 = f.urhs(u1,v1) -  f.Gpx(p1)
        rhs_v1 = f.vrhs(u1,v1) -  f.Gpy(p1)

        # stage 2:
        # ---------
        u2, v2, p2, info = f.DIRK_S2(un, vn, pn,rhs_u1,rhs_v1,DIRK2)
        print('        number of function calls stage 2: ', info['nfev'])
        # time n+1
        #----------
        uhnp1 = un + b1 * dt * f.urhs(u1,v1) + b2 * dt * f.urhs(u2, v2)
        vhnp1 = vn + b1 * dt * f.vrhs(u1,v1) + b2 * dt * f.vrhs(u2, v2)

        unp1, vnp1, press, iter = f.ImQ(uhnp1, vhnp1, Coef, pn)

        time_end = time.clock()
        psol.append(press)
        cpu_time = time_end - time_start
        print('        cpu_time=',cpu_time)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        print('        Mass residual:',residual)

        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        print('len Usol:',len(usol))
        print('Courant Number=',1.42*dt/dx)
        iterations.append(iter)
        # # u and v num cell centered
        ucc = 0.5*(un[1:-1,2:] + un[1:-1,1:-1])
        vcc = 0.5*(vn[2:,1:-1] + vn[1:-1,1:-1])
        #
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
                stability_counter += 1
        else:
            is_stable = True
            print('is_stable = ', is_stable)
            if ken_new < target_ke and count > 30:
                break
        ken_old = ken_new.copy()
        print('is_stable = ', is_stable)
        max = 4
        min = -4
        levels = np.linspace(min, max, 50, endpoint=True)
        # #plot of the pressure gradient in order to make sure the solution is correct
        # im = plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx,cmap='viridis',levels=levels,vmin=-4,vmax=4)
        # v = np.linspace(-4, 4, 5, endpoint=True)
        # cbar = plt.colorbar(im)
        # plt.title("time={:0.4f}s".format(t))
        # plt.tight_layout()
        # plt.savefig('Implicit-NSE/DIRK2_{}_capuano_form/animations/dpdx/timestep-{:0>2}.png'.format(name,count), dpi=300)
        # plt.close()

        count+=1
    diff = np.linalg.norm(uexact(a,b,xu,yu,t).ravel()-unp1[1:-1,1:] .ravel(),np.inf)
    if return_stability:
        return is_stable
    else:
        return diff, [total_iteration], is_stable, unp1[1:-1, 1:].ravel()


