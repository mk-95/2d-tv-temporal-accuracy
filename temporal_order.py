import numpy as np
import json
import matplotlib.pyplot as plt
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
from normal_velocity_bcs_RK3 import error_normal_velocity_bcs_RK3

from taylor_vortex_with_time_dependent_bcs_RK2 import error_tv_time_dependent_bcs_RK2
from taylor_vortex_with_time_dependent_bcs_RK3 import error_tv_time_dependent_bcs_RK3


# taylor vortex
#---------------
probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)


# # channel flow
# #--------------
# ν = 0.1
# Uinlet = 1
# probDescription = ProbDescription(N=[4*32,32],L=[4,1],μ =ν,dt = 0.05)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)


levels = 8        # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]

timesteps = [5*rt**i for i in range(0,levels) ]

# to run multiple cases
#========================
#RK2 integrators
#-----------------
stages_projections = [[0],[0],[1]]
guesses = [None,'first',None]
keys = ['RK20*','RK20','RK21']
integrator_name = 'heun'
theta = None

# # RK3 integrators
# #-----------------
# stages_projections = [[0,0],[0,0],[0,0],[0,1],[1,0],[1,1]]
# guesses = [None,'first','second','second','second',None]
# keys = ['RK300*','RK300**','RK300','RK301','RK310','RK311']
# integrator_name = 'heun'
# theta = None

# to run single case
#========================
# # RK2 integrator
# #----------------
# stages_projections = [[1]]
# guesses = [None]
# keys = ['RK21']
# integrator_name = 'heun'
# theta = None

# # RK3 integrator
# #----------------
# stages_projections = [[1,1]]
# guesses = [None]
# keys = ['RK311']
# integrator_name = 'heun'
# theta = None

file_name = 'tv_unsteady_bcs_{}.json'.format('RK2')
directory = './temporal_orders/taylor_vortex_unsteady_bcs/'

dict = {}
for proj, guess, key in zip(stages_projections,guesses,keys):
    phiAll = []
    for dt, nsteps in zip(dts, timesteps):
        probDescription.set_dt(dt)

        # taylor vortex
        #---------------
        # e, divs, _, phi =error_RK2(steps = nsteps,name=integrator_name,guess=guess,project=proj)
        # e, divs, _, phi = error_RK3(steps=nsteps, name='regular', guess='second', project=[0, 0])
        # e, divs, _, phi =error_RK4(steps = nsteps,name='3/8',guess=None,project=[1,1,1])

        # FE channel flow
        #-----------------
        # e, divs, _, phi =error_channel_flow_FE(steps=nsteps)
        # e, divs, _, phi =error_channel_flow_FE_unsteady_inlet(steps=nsteps)

        # RK2 channel flow
        # -----------------
        # e, divs, _, phi =error_channel_flow_RK2(steps = nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)
        # e, divs, _, phi = error_channel_flow_RK2_unsteady_inlet(steps = nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)

        # RK3 channel flow
        # -----------------
        # e, divs, _, phi = error_channel_flow_RK3(steps = nsteps,name=integrator_name,guess=guess,project=proj)
        # e, divs, _, phi = error_channel_flow_RK3_unsteady_inlet(steps = nsteps,name=integrator_name,guess=guess,project=proj)

        # lid driven cavity
        #-------------------
        # e, divs, _, phi =error_lid_driven_cavity_RK2(steps = nsteps,name='heun',guess=None,project=[1])
        # e, divs, _, phi = error_lid_driven_cavity_RK3(steps=nsteps, name='heun', guess=None, project=[1,1])

        # Poisseille flow
        #-----------------
        # e, divs, _, phi =error_normal_velocity_bcs_RK2(steps = nsteps,name=integrator_name,guess=guess,project=proj)
        # e, divs, _, phi =error_normal_velocity_bcs_RK3(steps = nsteps,name=integrator_name,guess=guess,project=proj)

        # unsteady boundary conditions taylor vortex
        #-------------------------------------------
        e, divs, _, phi = error_tv_time_dependent_bcs_RK2(steps=nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)
        # e, divs, _, phi = error_tv_time_dependent_bcs_RK3(steps=nsteps,name=integrator_name,guess=guess,project=proj)

        phiAll.append(phi)

    # local errors
    exact_err_mom = []
    errAll = []
    for i in range(0, levels - 1):
        diff = phiAll[i + 1] - phiAll[i]
        err = np.linalg.norm(diff, 2)
        print('error', err)
        errAll.append(err)

    # now compute order
    Order = []
    for i in range(0, levels - 2):
        Order.append(np.log(errAll[i + 1] / errAll[i]) / np.log(1.0 / rt))
        print('order: ', Order[-1])

    dict[key]={'dt':dts[:-1],"error": errAll}

with open(directory+file_name,"w") as file:
    json.dump(dict,file,indent=4)