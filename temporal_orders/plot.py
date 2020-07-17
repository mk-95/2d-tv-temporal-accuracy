import numpy as np
import json
import matplotlib.pyplot as plt

# RK2 channel flow
# file_name = 'channel_flow_unsteady_inlet_RK2.json'
file_name = 'channel_flow_steady_RK2.json'

# RK3 channel flow
# file_name = 'channel_flow_unsteady_inlet_RK3.json'
# file_name = 'channel_flow_steady_RK3.json'


with open(file_name,'r') as file:
    data = json.load(file)


fig_temp = plt.figure()

for key in data.keys():
    dts = np.array(data[key]['dt'])
    plt.loglog(dts, data[key]['error'], 'o-', label='Error')
    plt.loglog(dts,  0.5e3*dts**3, '--', label=r'$3^{rd}$')
    plt.loglog(dts,  1e2*dts**2, '--', label=r'$2^{nd}$')
    plt.loglog(dts,  0.5e4*dts**4, '--', label=r'$4^{th}$')
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$L_\infty$ Norm')
# plt.legend()
plt.show()