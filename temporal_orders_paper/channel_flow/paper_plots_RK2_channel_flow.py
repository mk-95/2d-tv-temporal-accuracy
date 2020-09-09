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
with open('channel_flow_steady_RK2.json') as f:
    data = json.load(f)

for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK20':
        offset = 1.5
    ax1.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

with open('channel_flow_unsteady_inlet_RK2.json') as f:
    data = json.load(f)
for key, c, m, lbl in zip(data.keys(), colors, markers, labels):
    dt = np.array(data[key]['dt'])
    err = np.array(data[key]['error'])
    offset = 1
    if key == 'RK20':
        offset = 1.5
    ax2.loglog(dt, offset*err, color=c, linewidth=1.5, marker=m, markersize=4, label=lbl)

dt = np.array([7e-4, 3e-3])
ax1.loglog(dt, 1.25 * dt, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t)$', xy=(1.5e-3, 9e-4))
ax1.loglog(dt, 45 * dt ** 2, '--k', linewidth=1)
ax1.annotate('$\mathcal{O}(\Delta t^2)$', xy=(1.5e-3, 7e-5))

dt = np.array([7e-5, 3e-4])
ax2.loglog(dt, 0.5e1 * dt, '--k', linewidth=1)
# ax2.annotate('$\mathcal{O}(\Delta t)$', xy=(1.5e-3, 9e-4))
ax2.loglog(dt, 1e3 * dt ** 2, '--k', linewidth=1)
# ax2.annotate('$\mathcal{O}(\Delta t^2)$', xy=(1.5e-3, 7e-5))

ax1.set_ylabel('$L_\infty$', fontsize=10)
ax1.set_xlabel('$\Delta t$', fontsize=10)
ax1.set_title('Steady BCs', fontsize=9)

ax2.legend(loc='lower right', fontsize=8, frameon=False)
ax2.set_xlabel(r'$\Delta t$', fontsize=10)
ax2.set_title('Unsteady BCs', fontsize=9)

plt.tight_layout()
plt.savefig('rk2-channel-flow-steady-and-unsteady-bcs-convergence-rate.pdf')
plt.show()