import numpy as np

# from error_func_FE import *
# from capuanos import *
# from error_func_capuano import error_capuano
import matplotlib.pyplot as plt
import json
from singleton_classes import ProbDescription


# from error_func_RK2 import error_RK2
# from error_func_RK3 import error_RK3
from error_func_RK4 import error_RK4

from error_func_RK2_with_post_projection import error_RK2_with_post_projection
from error_func_RK3_with_post_projection import error_RK3_with_post_projection

from lid_driven_cavity_FE import error_lid_driven_cavity_FE
from lid_driven_cavity_RK2 import error_lid_driven_cavity_RK2
from lid_driven_cavity_RK3 import error_lid_driven_cavity_RK3
# from lid_driven_cavity_RK3 import error_lid_driven_cavity_RK3

# from channel_flow_FE import error_channel_flow_FE
# from channel_flow_FE_unsteady_inlet import error_channel_flow_FE_unsteady_inlet

from channel_flow_RK2 import error_channel_flow_RK2
from channel_flow_RK2_unsteady_inlet import error_channel_flow_RK2_unsteady_inlet

from error_func_RK2_with_post_projection import error_RK2_with_post_projection
from channel_flow_RK2_with_post_projection import error_channel_flow_RK2_with_post_projection

from channel_flow_RK3 import error_channel_flow_RK3
from channel_flow_RK3_unsteady_inlet import error_channel_flow_RK3_unsteady_inlet


from normal_velocity_bcs import error_normal_velocity_bcs_RK2

from taylor_vortex_with_time_dependent_bcs import error_tv_time_dependent_bcs_FE
from taylor_vortex_with_time_dependent_bcs_RK2 import error_tv_time_dependent_bcs_RK2
from taylor_vortex_with_time_dependent_bcs_RK3 import error_tv_time_dependent_bcs_RK3

from channel_flow_RK4 import error_channel_flow_RK4

# # taylor vortex
# #---------------
# probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)

# lid-driven-cavity
#-------------------
# ν = 0.01
# Uinlet = 1
# probDescription = ProbDescription(N=[16,16],L=[1,1],μ =ν,dt = 0.005)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)

# channel flow
# --------------
ν = 0.1
Uinlet = 1
probDescription = ProbDescription(N=[4*32,32],L=[4,1],μ =ν,dt = 0.05)
dx,dy = probDescription.dx, probDescription.dy
dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
probDescription.set_dt(dt/4)


levels = 5        # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]

# dts_pp = [1.1*probDescription.get_dt()/rt**i for i in range(0,levels)]
# dts_pp = [1e-1/rt**i for i in range(0,levels)]
dts_pp = [1e-4 for i in range(0,levels)]
timesteps = [5*rt**i for i in range(0,levels) ]
print(timesteps)
Dginv = 0
Divergence = {}
Errors =[]
num=0
print('dts=',dts)

print ('---------------- TEMPORAL ORDER -------------------')
phiAll = []
probDescription.rt = 1
for dt, nsteps,new_dt in zip(dts, timesteps,dts_pp):
    probDescription.set_dt(dt)
    probDescription.dt_post_processing = new_dt
    # taylor vortex
    #---------------
    # e, divs, _, phi =error_RK2(steps = nsteps,name='midpoint',guess='first',project=[0])
    # e, divs, _, phi = error_RK3(steps=nsteps, name='regular', guess='second', project=[0, 0])
    # e, divs, _, phi = error_RK3_with_post_projection(steps=nsteps, name='regular', guess='second', project=[1, 0],post_projection=True)
    # e, divs, _, phi = error_RK2_with_post_projection(steps=nsteps, name='heun', guess='ml_dt', project=[0],post_projection=False,ml_model='ML/model4_siren_dt/best.json',ml_weights='ML/model4_siren_dt/best.h5')
    # e, divs, _, phi =error_RK4(steps = nsteps,name='3/8',guess="post-processing-approx",project=[0,0,0])

    # FE channel flow
    #-----------------
    # e, divs, _, phi =error_channel_flow_FE(steps=nsteps)
    # e, divs, _, phi =error_channel_flow_FE_unsteady_inlet(steps=nsteps)

    # RK2 channel flow
    # -----------------
    # e, divs, _, phi =error_channel_flow_RK2(steps = nsteps,name='theta',guess='first',project=[1],theta=0.25)
    # e, divs, _, phi = error_channel_flow_RK2_unsteady_inlet(steps=nsteps, name='theta', guess='first', project=[0],theta=0.25)

    # RK2 post processing pressure
    #-------------------------------------------
    # e, divs, _, phi = error_RK2_with_post_projection(steps=nsteps, name='heun', guess='first', project=[0])
    # e, divs, _, phi = error_channel_flow_RK2_with_post_projection(steps=nsteps, name='theta', guess="first", project=[0], theta=0.5)

    # RK3 channel flow
    # -----------------
    # e, divs, _, phi = error_channel_flow_RK3(steps=nsteps, name='heun', guess=None, project=[1,1])
    e, divs, _, phi = error_channel_flow_RK3_unsteady_inlet(steps=nsteps, name='regular', guess=None,project=[1, 1])

    # RK4 channel flow
    # -----------------
    # e, divs, _, phi = error_channel_flow_RK4(steps=nsteps, name='3/8', guess="post-processing-approx", project=[0,0,0])
    # e, divs, _, phi = error_channel_flow_RK4_unsteady_inlet(steps=nsteps, name='regular', guess='second',project=[0, 1])

    # lid driven cavity
    #-------------------
    # e, divs, _, phi =error_lid_driven_cavity_RK2(steps = nsteps,name='heun',guess=None,project=[1])
    # e, divs, _, phi = error_lid_driven_cavity_RK3(steps=nsteps, name='heun', guess=None, project=[1,1])

    # Poisseille flow
    #-----------------
    # e, divs, _, phi =error_normal_velocity_bcs_RK2(steps = nsteps,name='theta',guess=None,project=[1],theta=0.1)

    # e, divs, _, phi = error_tv_time_dependent_bcs_FE(steps=nsteps)
    # e, divs, _, phi =error_tv_time_dependent_bcs_RK2(steps = nsteps,name='theta',guess='first',project=[0],theta=0.25)
    # e, divs, _, phi =error_tv_time_dependent_bcs_RK3(steps = nsteps,name='heun',guess='second',project=[0,0])

    phiAll.append(phi)
# local errors
exact_err_mom=[]
errAll = []
for i in range(0,levels-1):
    diff = phiAll[i+1] - phiAll[i]
    err = np.linalg.norm(diff,np.inf)
    print ('error', err)
    errAll.append(err)

# now compute order
Order=[]
for i in range(0,levels-2):
    Order.append(np.log( errAll[i+1]/errAll[i] ) / np.log(1.0/rt))
    print ('order: ',Order[-1])


# dict = {'timesteps':dts,'x-mom':{"order":Order, "error":errAll}}
# # save the values in a json file
# name = "post-processing-pressure/unsteady-rk3-1e-4.txt"
# with open(name,"w") as file:
#     json.dump(dict,file,indent=4)


fig_temp = plt.figure()
plt.loglog(np.array(dts)[:-1], errAll, 'o-', label='Error')
plt.loglog(np.array(dts),  0.5e5*np.array(dts)**3, '--', label=r'$3^{rd}$')
plt.loglog(np.array(dts),  1e4*np.array(dts)**2, '--', label=r'$2^{nd}$')
# plt.loglog(np.array(dts),  0.5e4*np.array(dts)**4, '--', label=r'$4^{th}$')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$L_\infty$ Norm')
plt.legend()
plt.show()
