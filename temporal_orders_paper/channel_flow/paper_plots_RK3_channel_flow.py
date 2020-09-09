import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

import matplotlib

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
colors = ['k', '#2E7990', '#A43F45', '#915F56', '#C49B4C', '#606c38']
markers = ['o', 'o', 'o', 's', 'D', 'v']
labels = ['RK3$_{00}^*$', 'RK3$_{00}^{**}$', 'RK3$_{00}$', 'RK3$_{01}$', 'RK3$_{10}$', 'RK3$_{11}$']
with open('channel_flow_steady_RK3.json') as f:
    data = json.load(f)

for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK300':
        offset = 2
    elif key == 'RK301':
        offset = 1.5
    ax1.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

with open('channel_flow_unsteady_inlet_RK3.json') as f:
    data = json.load(f)
for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK300':
        offset = 2
    elif key == 'RK301':
        offset = 1.5
    ax2.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

dt = np.array([7e-4, 3e-3])
ax1.loglog(dt, 1 * dt, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t)$', xy=(1e-3, 5e-4))
ax1.loglog(dt, 7 * dt ** 2, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t^2)$', xy=(0.85e-3, 3e-6))
ax1.loglog(dt, 90 * dt ** 3, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t^3)$', xy=(1e-3, 5e-8))

ax2.loglog(dt, 1 * dt, '--k', linewidth=1)
# ax2.annotate('$\mathcal{O}(\Delta t)$', xy=(1e-3, 5e-4))
ax2.loglog(dt, 6 * dt ** 2, '--k', linewidth=1)
# ax2.annotate('$\mathcal{O}(\Delta t^2)$', xy=(1e-3, 3e-6))
ax2.loglog(dt, 150 * dt ** 3, '--k', linewidth=1)
# ax2.annotate('$\mathcal{O}(\Delta t^3)$', xy=(1e-3, 5e-8))

ax1.set_ylabel('$L_\infty$', fontsize=10)
ax1.set_xlabel('$\Delta t$', fontsize=10)
ax1.set_title('Steady BCs', fontsize=10)

ax2.legend(ncol=2, loc='lower right', fontsize=7, frameon=False, borderpad=0.01)
ax2.set_xlabel(r'$\Delta t$', fontsize=10)
ax2.set_title('Unsteady BCs', fontsize=10)
ax2.set_ylim([1e-9, 1e-2])
# ax1.set_ylim([1e-9, 1.5e-2])
ax2.minorticks_on()
plt.tight_layout()
plt.savefig('rk3-channel-flow-stead-and-unsteady-bcs-convergence-rate.pdf')
plt.show()