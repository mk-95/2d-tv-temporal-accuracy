import numpy as np

# from error_func_FE import *
# from capuanos import *
# from error_func_capuano import error_capuano
import matplotlib.pyplot as plt
import json
from singleton_classes import ProbDescription
from error_func_RK2 import error_RK2
from error_func_RK3 import error_RK3
from error_func_RK4 import error_RK4


probDescription = ProbDescription(N=[32,32],L=[1,1],μ =1e-3,dt = 0.005)


levels = 8        # total number of refinements

rx = 1
ry = 1
rt = 2

dts = [probDescription.get_dt()/rt**i for i in range(0,levels)]

timesteps = [10*rt**i for i in range(0,levels) ]

Dginv = 0
Divergence = {}
Errors =[]
num=0
print('dts=',dts)

print ('---------------- TEMPORAL ORDER -------------------')
phiAll = []
for dt, nsteps in zip(dts, timesteps):
    probDescription.set_dt(dt)
    # e, divs, _, phi =error_RK2(steps = nsteps,name='midpoint',guess='first',project=[0])
    # e, divs, _, phi = error_RK3(steps=nsteps, name='regular', guess='second', project=[1, 1])
    e, divs, _, phi =error_RK4(steps = nsteps,name='3/8',guess='fourth',project=[0,0,1])
    # e, divs, _, phi =error_capuanos(dt=dt,μ=μ,n=Nx,steps = nsteps,guess='capuano',project=[1,0])
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
    Order.append(np.log( errAll[i+1]/errAll[i] ) / np.log(0.5))
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