import numpy as np

# from error_func_FE import *
# from capuanos import *
# from error_func_capuano import error_capuano
import matplotlib.pyplot as plt
import json
from singleton_classes import ProbDescription


# from error_func_RK2 import error_RK2
# from error_func_RK3 import error_RK3
# from error_func_RK4 import error_RK4

from lid_driven_cavity_FE import error_lid_driven_cavity_FE
from lid_driven_cavity_RK2 import error_lid_driven_cavity_RK2
# from lid_driven_cavity_RK3 import error_lid_driven_cavity_RK3

# from channel_flow_FE import error_channel_flow_FE
# from channel_flow_FE_unsteady_inlet import error_channel_flow_FE_unsteady_inlet

from channel_flow_RK2 import error_channel_flow_RK2
from channel_flow_RK2_unsteady_inlet import error_channel_flow_RK2_unsteady_inlet

from channel_flow_RK3 import error_channel_flow_RK3
from channel_flow_RK3_unsteady_inlet import error_channel_flow_RK3_unsteady_inlet


from normal_velocity_bcs import error_normal_velocity_bcs_RK2

from taylor_vortex_with_time_dependent_bcs import error_tv_time_dependent_bcs_FE
from taylor_vortex_with_time_dependent_bcs_RK2 import error_tv_time_dependent_bcs_RK2
from taylor_vortex_with_time_dependent_bcs_RK3 import error_tv_time_dependent_bcs_RK3

# taylor vortex
#---------------
probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)

# lid-driven-cavity
#-------------------
# ν = 0.01
# Uinlet = 1
# probDescription = ProbDescription(N=[16,16],L=[1,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)

# channel flow
#--------------
# ν = 0.1
# Uinlet = 1
# probDescription = ProbDescription(N=[4*8,8],L=[4,1],μ =ν,dt = 0.05)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)


levels = 7        # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]

timesteps = [5*rt**i for i in range(0,levels) ]

print(timesteps)
Dginv = 0
Divergence = {}
Errors =[]
num=0
print('dts=',dts)

print ('---------------- TEMPORAL ORDER -------------------')
phiAll = []
for dt, nsteps in zip(dts, timesteps):
    probDescription.set_dt(dt)

    # taylor vortex
    #---------------
    # e, divs, _, phi =error_RK2(steps = nsteps,name='midpoint',guess='first',project=[0])
    # e, divs, _, phi = error_RK3(steps=nsteps, name='regular', guess='second', project=[0, 0])
    # e, divs, _, phi =error_RK4(steps = nsteps,name='3/8',guess=None,project=[1,1,1])

    # FE channel flow
    #-----------------
    # e, divs, _, phi =error_channel_flow_FE(steps=nsteps)
    # e, divs, _, phi =error_channel_flow_FE_unsteady_inlet(steps=nsteps)

    # RK2 channel flow
    # -----------------
    # e, divs, _, phi =error_channel_flow_RK2(steps = nsteps,name='theta',guess='first',project=[0],theta=0.25)
    # e, divs, _, phi = error_channel_flow_RK2_unsteady_inlet(steps=nsteps, name='theta', guess='first', project=[0],theta=0.25)

    # RK3 channel flow
    # -----------------
    # e, divs, _, phi = error_channel_flow_RK3(steps=nsteps, name='heun', guess=None, project=[1,1])
    # e, divs, _, phi = error_channel_flow_RK3_unsteady_inlet(steps=nsteps, name='regular', guess='second',project=[0, 1])

    # lid driven cavity
    #-------------------
    # e, divs, _, phi =error_lid_driven_cavity_RK2(steps = nsteps,name='heun',guess=None,project=[1])
    # e, divs, _, phi = error_lid_driven_cavity_RK3(steps=nsteps, name='heun', guess=None, project=[1,1])

    # Poisseille flow
    #-----------------
    # e, divs, _, phi =error_normal_velocity_bcs_RK2(steps = nsteps,name='theta',guess=None,project=[1],theta=0.1)

    # e, divs, _, phi = error_tv_time_dependent_bcs_FE(steps=nsteps)
    # e, divs, _, phi =error_tv_time_dependent_bcs_RK2(steps = nsteps,name='theta',guess='first',project=[0],theta=0.25)
    e, divs, _, phi =error_tv_time_dependent_bcs_RK3(steps = nsteps,name='heun',guess='second',project=[0,0])

    phiAll.append(phi)
# local errors
exact_err_mom=[]
errAll = []
for i in range(0,levels-1):
    diff = phiAll[i+1] - phiAll[i]
    err = np.linalg.norm(diff,2)
    print ('error', err)
    errAll.append(err)

# now compute order
Order=[]
for i in range(0,levels-2):
    Order.append(np.log( errAll[i+1]/errAll[i] ) / np.log(1.0/rt))
    print ('order: ',Order[-1])


dict = {'timesteps':dts,'x-mom':{"order":Order, "error":errAll}}
# save the values in a json file
name = "./s1_3_s2_p_s3_p.txt"
# with open(name,"w") as file:
#     json.dump(dict,file,indent=4)


fig_temp = plt.figure()
plt.loglog(np.array(dts)[:-1], errAll, 'o-', label='Error')
plt.loglog(np.array(dts),  0.5e3*np.array(dts)**3, '--', label=r'$3^{rd}$')
plt.loglog(np.array(dts),  1e2*np.array(dts)**2, '--', label=r'$2^{nd}$')
plt.loglog(np.array(dts),  0.5e4*np.array(dts)**4, '--', label=r'$4^{th}$')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$L_\infty$ Norm')
plt.legend()
plt.show()
