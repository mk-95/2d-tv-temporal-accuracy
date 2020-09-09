import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

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

ax1.grid(which='minor', alpha=0.19)
ax2.grid(which='minor', alpha=0.19)
colors = ['k', '#2E7990', '#A43F45', '#915F56', '#C49B4C']
markers = ['o', 's', 'D', 'v']
labels = ['RK2$_0^*$', 'RK2$_1$', 'RK2$_0$']
with open('tv_unsteady_bcs_RK2.json') as f:
    data = json.load(f)

for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK20':
        offset = 1.5
    ax1.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

with open('tv_unsteady_bcs_RK3.json') as f:
    data = json.load(f)

colors = ['k','#2E7990','#A43F45','#915F56','#C49B4C','#606c38']
markers= ['o','o','o','s','D','v']
labels=['RK3$_{00}^*$','RK3$_{00}^{**}$','RK3$_{00}$','RK3$_{01}$','RK3$_{10}$','RK3$_{11}$']

with open('tv_unsteady_bcs_RK3.json') as f:
  data = json.load(f)

for key,c,m,lbl in zip(data.keys(),colors,markers,labels):
    dt = data[key]['dt']
    err = data[key]['error']
    ax2.loglog(dt,err,color=c,linewidth=1.5,marker=m,markersize=4,label=lbl)

dt = np.array([21e-5, 12e-4])
ax1.loglog(dt, 2 * dt, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t)$', xy=(4e-4, 4e-4))
ax1.loglog(dt, 45 * dt ** 2, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t^2)$', xy=(4e-4, 4e-6))

dt = np.array([7e-5,3e-4])
ax2.loglog(dt,1*dt,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t)$',xy=(1e-4,2e-5))
ax2.loglog(dt,10*dt**2,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t^2)$',xy=(1e-4,2e-8))
ax2.loglog(dt,25*dt**3,'--k',linewidth=1)
ax2.annotate('$\mathcal{O}(\Delta t^3)$',xy=(1e-4,0.5e-11))


ax1.legend(loc='lower right',fontsize=7,frameon=False)
ax1.set_ylabel('$L_\infty$', fontsize=10)
ax1.set_xlabel('$\Delta t$', fontsize=10)
# ax1.set_title('Unsteady Bounadry Conditions RK2\n2D Taylor Vortex ', fontsize=9)

ax2.legend(ncol=2,loc='lower right',fontsize=7,frameon=False, columnspacing=1.0)
ax2.set_xlabel(r'$\Delta t$',fontsize=10)
# ax2.set_title('Unsteady BCs RK3', fontsize=9)
ax2.set_ylim([1e-12,1e-1])

plt.tight_layout()
plt.savefig('rk2-and-rk3-taylor-vortex-unsteady-bcs-convergence-rate.pdf')
plt.show()