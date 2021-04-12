import numpy as np
from functions import func
import time
import json
import singleton_classes as sc
from analytical_jacobian import Analytic_Jacobian
from sym_residual_functions import SymResidualFunc
from Jacobian_indexing import PeriodicIndexer


def error_BE_analytical_jacobian (steps=3, return_stability=False,alpha=0.99):
    # problem description
    probDescription = sc.ProbDescription()
    dt = probDescription.get_dt()
    # symbolic residual functions constructions
    Res_funcs = SymResidualFunc(probDescription)
    lhs_u, lhs_v, lhs_p = Res_funcs.lhs()
    rhs_u, rhs_v, rhs_p = Res_funcs.rhs()

    # unsteady residual
    #-------------------
    sym_f1 = dt*(lhs_u - rhs_u)
    sym_f2 = dt*(lhs_v - rhs_v)
    sym_f3 = dt*(lhs_p - rhs_p)


    x1,x2,x3 = Res_funcs.vars()

    j, i = Res_funcs.indices()

    stencil_pts = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
    # Jacobian Builder for every stage
    Jacobian_builder = Analytic_Jacobian([sym_f1,sym_f2,sym_f3],[x1,x2,x3],[j,i],[-1,1],stencil_pts,probDescription,PeriodicIndexer)

    f = func(probDescription, 'periodic')

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

    pexact = lambda x, y, t: (-8 * np.sin(np.pi * t) ** 4 * np.sin(np.pi * y) ** 4 - 2 * np.sin(
        np.pi * t) ** 4 - 2 * np.sin(np.pi * y) ** 4 - 5 * np.cos(
        2 * np.pi * t) / 2 + 5 * np.cos(4 * np.pi * t) / 8 - 5 * np.cos(2 * np.pi * y) / 2 + 5 * np.cos(
        4 * np.pi * y) / 8 - np.cos(
        np.pi * (2 * t - 4 * y)) / 4 + np.cos(np.pi * (2 * t - 2 * y)) + np.cos(np.pi * (2 * t + 2 * y)) - np.cos(
        np.pi * (2 * t + 4 * y)) / 4 - 3 * np.cos(np.pi * (4 * t - 4 * y)) / 16 - np.cos(
        np.pi * (4 * t - 2 * y)) / 4 - np.cos(
        np.pi * (4 * t + 2 * y)) / 4 + np.cos(np.pi * (4 * t + 4 * y)) / 16 + 27 / 8) * np.exp(
        -16 * np.pi ** 2 * μ * t) - np.exp(
        -16 * np.pi ** 2 * μ * t) * np.cos(np.pi * (-4 * t + 4 * x)) / 4
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

    # # for debugging the augment_fields_size_NSE
    # #-----------
    # new_u0,new_v0 = Jacobian_builder.augment_fields_size_NSE(u0,v0)
    #
    # plt.imshow(new_u0,origin='bottom')
    # plt.show()
    # #-----------

    # initialize the pressure
    p0 = np.zeros([nx+2,ny+2]); # include ghost cells
    p0[1:-1, 1:-1] = pexact(xcc, ycc, 0)
    f.periodic_scalar(p0)

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

    info_resid = {}

    while count < tend:
        print('timestep:{}'.format(count+1))

        time_start = time.clock()
        un = usol[-1].copy()
        vn = vsol[-1].copy()
        pn = psol[-1].copy()

        # unsteady residual
        #-------------------
        f1 = lambda uold,vold,pold: lambda u,v,p: dt*((u - uold) / dt - (f.urhs(u, v) - f.Gpx(p)))
        f2 = lambda uold,vold,pold: lambda u,v,p: dt*((v - vold) / dt - (f.vrhs(u, v) - f.Gpy(p)))
        # f3 = lambda uold, vold: lambda u,v,p: (f.laplacian(p) - f.div(f.urhs(u, v),
        #                                                     f.vrhs(u, v))) - f.div(uold, vold) / dt
        f3 = lambda uold,vold,pold: lambda u, v, p: dt*((f.div(f.Gpx(p),f.Gpy(p)) - f.div(f.urhs(u, v),
                                                                        f.vrhs(u, v))) - f.div(uold, vold) / dt)



        old = [un,vn,pn]
        residuals = [f1,f2,f3]
        guesses = [un,vn,pn]
        Tol = 1e-8
        sol,iterations,error, info = f.Newton_solver(guesses,old, residuals, [f.periodic_u,f.periodic_v,f.periodic_scalar],Jacobian_builder, Tol,True)
        info_resid[count]=info
        unp1, vnp1, press = sol
        f.periodic_u(unp1)
        f.periodic_v(vnp1)
        f.periodic_scalar(press)

        # print('         number of function calls: ', info['nfev'])

        time_end = time.clock()
        psol.append(press)
        cpu_time = time_end - time_start
        print('Courant Number=', 1.42 * dt / dx)
        print('        cpu_time=', cpu_time)
        print('        non-linear iterations=', iterations)
        print('        error=', error)
        # Check mass residual
        div_np1 = np.linalg.norm(f.div(unp1, vnp1).ravel())
        residual = div_np1
        #         if residual > 1e-12:
        print('        Mass residual:',residual)
        # save new solutions
        usol.append(unp1)
        vsol.append(vnp1)
        print('len Usol:',len(usol))
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
        max = 2
        min = 0
        levels = np.linspace(min, max, 20, endpoint=True)
        #plot of the pressure gradient in order to make sure the solution is correct
        # im = plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx,cmap='viridis',levels=levels,vmin=min,vmax=max)
        # im = plt.contourf(f.div(unp1,vnp1)[1:-1,1:] ,cmap='viridis')
        im = plt.imshow(usol[-1][1:-1,1:])
        # im = plt.contourf(f.div(usol[-1],vsol[-1])[1:-1,1:-1],cmap='viridis',levels=levels,vmin=0.0,vmax=2.0)
        # plt.contourf((psol[-1][1:-1,1:] - psol[-1][1:-1,:-1])/dx)
        v = np.linspace(min, max, 5, endpoint=True)
        cbar = plt.colorbar(im)
        # cbar.set_ticks(v)
        plt.title("time={:0.4f}s".format(t))
        plt.tight_layout()
        # plt.show()
        plt.savefig('analytical_J_Implicit_NSE/Backward_Euler/test-convergence/Pyamg/CFL-0.1/unp1/timestep-{:0>2}.png'.format(count),dpi = 400)
        plt.close()
        count+=1
    diff = np.linalg.norm(uexact(a,b,xu,yu,t).ravel()-unp1[1:-1,1:] .ravel(),np.inf)
    # print('        error={}'.format(diff))
    if return_stability:
        return is_stable
    else:
        return diff, [total_iteration], is_stable, unp1[1:-1, 1:].ravel(), info_resid


from singleton_classes import ProbDescription
import matplotlib.pyplot as plt
#
dt_lam = lambda CFL, dx,Uinlet: CFL*dx/Uinlet
Uinlet = 1.42
# ν = 0.1
ν = 0.2
probDescription = ProbDescription(N=[24,24],L=[1,1],μ =ν,dt = 0.01)
dx,dy = probDescription.dx, probDescription.dy

dt = dt_lam(0.1,dx,Uinlet)
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
probDescription.set_dt(dt)

_,_,_,_,resid_info = error_BE_analytical_jacobian(steps=20, return_stability=False,alpha=0.99)

# resid_filename ='analytical_J_Implicit_NSE/Backward_Euler/test-convergence/Pyamg/CFL-0.1/residuals_per_timestep.json'
# with open(resid_filename,"w") as file:
#     json.dump(resid_info,file,indent=4)