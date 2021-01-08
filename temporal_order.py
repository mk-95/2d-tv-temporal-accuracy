import numpy as np
import json
import matplotlib.pyplot as plt
from singleton_classes import ProbDescription

from error_func_RK2 import error_RK2
from error_func_RK3 import error_RK3
from error_func_RK4 import error_RK4
from error_func_capuano import error_capuanos

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

from taylor_vortex_with_steady_bcs_RK2 import error_tv_steady_bcs_RK2

# taylor vortex
# ---------------
probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.0025)


# # channel flow steady
# #--------------------
# ν = 0.1
# Uinlet = 1
# probDescription = ProbDescription(N=[4*8,8],L=[4,1],μ =ν,dt = 0.05)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)
# probDescription.set_dt(1e-4)


# channel flow unsteady_inlet
#-----------------------------
# ν = 0.1
# Uinlet = 1
# probDescription = ProbDescription(N=[4*8,8],L=[4,1],μ =ν,dt = 0.05)
# # probDescription = ProbDescription(N=[4*32,32],L=[10,1],μ =ν,dt = 0.05)
# dx,dy = probDescription.dx, probDescription.dy
# dt = min(0.25*dx*dx/ν,0.25*dy*dy/ν, 4.0*ν/Uinlet/Uinlet)
# probDescription.set_dt(dt/4)
# # probDescription.set_dt(1e-4)

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
integrator_name = 'theta'
theta = 2.0/3

# # RK3 integrators
# #-----------------
# stages_projections = [[0,0],[0,0],[0,0],[0,1],[1,0],[1,1]]
# guesses = [None,'first','second','second','second',None]
# keys = ['RK300*','RK300**','RK300','RK301','RK310','RK311']
# integrator_name = 'heun'
# theta = None

# # RK4 integrators
# #-----------------
# stages_projections = [[0,0,0]]
# guesses = ["third"]
# keys = ['RK4000']
# integrator_name = '3/8'
# theta = None

# to run single case
#========================
# # RK2 integrator
# # ----------------
# stages_projections = [[0]]
# guesses = ['first']
# keys = ['RK20']
# integrator_name = 'heun'
# theta = None

# # RK3 integrator
# #----------------
# stages_projections = [[1,1]]
# guesses = [None]
# keys = ['RK311']
# integrator_name = 'heun'
# theta = None

# # RK3 integrator Capuano
# #-----------------------
# stages_projections = [[0, 0],[1, 0], [0, 1],[1,1]]
# guesses = ['capuano_ci_00','capuano_ci_10','capuano_ci_01',None]
# # guesses = ['capuano_00','capuano_10','capuano_01',None]
# keys = ['RK300','RK310','RK301','RK311']
# # keys = ['FC_ci_00','FC_ci_10','FC_ci_01']
# # keys = ['KS_ci_00','KS_ci_10','KS_ci_01']
# integrator_name = 'regular'
# theta = None

file_name = 'taylor_vortex_2D_pressure_atol_stage_2_1e-2_{}.json'.format('theta_raltson')
# file_name = 'channel_flow_steady_inlet_RK3_128_32_{}.json'.format('heun')
# file_name = 'taylor_vortex_2D_RK2_{}.json'.format('heun')
directory = './iteration-count/RK2-tv-2d/cheap-solve-intermediate-stage/temporal_order/'

dict = {}
for proj, guess, key in zip(stages_projections,guesses,keys):
    phiAll = []
    for dt, nsteps in zip(dts, timesteps):
        probDescription.set_dt(dt)

        # taylor vortex
        #---------------
        e, divs, _, phi =error_RK2(steps = nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)
        # e, divs, _, phi = error_RK3(steps=nsteps, name=integrator_name, guess=guess, project=proj)
        # e, divs, _, phi =error_RK4(steps = nsteps,name=integrator_name,guess=guess,project=proj)

        # Taylor Vortex Capuano
        # -----------------------
        # e, divs, _, phi = error_capuanos(steps=nsteps, name=integrator_name, guess=guess, project=proj)

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
        # e, divs, _, phi = error_tv_time_dependent_bcs_RK2(steps=nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)
        # e, divs, _, phi = error_tv_time_dependent_bcs_RK3(steps=nsteps,name=integrator_name,guess=guess,project=proj)

        # steady boundary conditions taylor vortex
        # -------------------------------------------
        # e, divs, _, phi = error_tv_steady_bcs_RK2(steps=nsteps,name=integrator_name,guess=guess,project=proj,theta=theta)
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