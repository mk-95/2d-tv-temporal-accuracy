import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # list of colors i used from matplotlib

font = {'family': 'serif',
        'weight': 'normal',
        'size': 9}

matplotlib.rc('font', **font)
matplotlib.rc('lines', lw=1)
matplotlib.rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'stix'
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
f.set_size_inches([6.5, 3.0])

axins = zoomed_inset_axes(ax2,2, bbox_to_anchor=[6.3e2,1.8e2,5,-7]) # zoom-factor: 2.5, location: upper-left

ax1.grid(which='minor', alpha=0.19)
ax2.grid(which='minor', alpha=0.19)
colors = ['k', '#2E7990', '#A43F45', '#915F56', '#C49B4C']
markers = ['o', 's', 'D', 'v']
labels = ['RK2$_0^*$', 'RK2$_1$', 'RK2$_0$']
with open('rk2-tv-3d-temp-order.json') as f:
    data = json.load(f)

for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK20':
        offset = 1.5
    ax1.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)


colors = ['k','#2E7990','#A43F45','#915F56','#C49B4C','#606c38']
markers= ['o','o','o','s','D','v']
labels=['RK3$_{00}^*$','RK3$_{00}^{**}$','RK3$_{00}$','RK3$_{01}$','RK3$_{10}$','RK3$_{11}$']

with open('rk3-tv-3D-temp-order.json') as f:
  data = json.load(f)

for key,c,m,lbl in zip(data.keys(),colors,markers,labels):
    dt = data[key]['dt']
    err = np.array(data[key]['error'])
    offset = 1
    ax2.loglog(dt,offset*err,color=c,linewidth=1.5,marker=m,markersize=4,label=lbl)
    if key != "RK300*" and key != "RK300**":
        if key == 'RK300':
            offset = 1
        elif key == 'RK301':
            offset = 1
        elif key == 'RK310':
            offset = 1
        axins.loglog(dt, offset * err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

x1, x2, y1, y2 = 1.15e-4, 1.4e-4, 3e-7, 4e-6  # specify the limits
axins.set_xlim(x1, x2)  # apply the x-limits
axins.set_ylim(y1, y2)  # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

dt = np.array([7e-5, 5e-4])
ax1.loglog(dt, 2 * dt, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t)$', xy=(1.5e-4, 1.5e-4))
ax1.loglog(dt, 90 * dt ** 2, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t^2)$', xy=(1.5e-4, 8e-7))

dt = np.array([3e-5,1.5e-4])
ax2.loglog(dt,1*dt,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t)$',xy=(5.5e-5,1.5e-5))
ax2.loglog(dt,25*dt**2,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t^2)$',xy=(5.5e-5,1.5e-8))
ax2.loglog(dt,140*dt**3,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t^3)$',xy=(5.5e-5,0.5e-11))


ax1.legend(loc='lower right',fontsize=7,frameon=False)
ax1.set_ylabel('$L_\infty$', fontsize=10)
ax1.set_xlabel('$\Delta t$', fontsize=10)
# ax1.set_title('Unsteady Bounadry Conditions RK2\n2D Taylor Vortex ', fontsize=9)

ax2.legend(ncol=2,loc='lower right',fontsize=7,frameon=False, columnspacing=1.0)
ax2.set_xlabel(r'$\Delta t$',fontsize=10)
# ax2.set_title('Unsteady BCs RK3', fontsize=9)
ax2.set_ylim([1e-12,1e-2])

plt.tight_layout()
# plt.savefig('taylor-green-vortex-3D-periodic-bcs-convergence-rate.pdf')
plt.show()