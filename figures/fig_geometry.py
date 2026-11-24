"""Fig. 1: the infalling clock in global AdS2 strip coordinates (theta, T); ingoing and outgoing labels. tau_s' = pi/4."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.linewidth': 0.6})
BLUE, ORANGE, INK, GREY = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8a86'
ts = np.pi/4
gd = lambda y: np.arctan(np.sinh(y))                                  # global time of the boundary label y
tau = lambda y: 2*np.arctan(np.tan(np.pi/2 - ts/2)*np.tanh(y/2))      # classical reading (proper time from the Phi = 0 event)
def clock(s):                                                          # sigma = 0: time-symmetric point; sigma = pi/2: Phi = 0
    return np.arctan(np.cos(s)/np.tan(ts)), np.arctan2(np.sin(ts)*np.sin(s), np.cos(s))
TOP, LEFT, BOT = 2.7, -0.95, -1.2
fig, ax = plt.subplots(figsize=(3.4, 3.9))
ax.plot([np.pi/2]*2, [BOT, TOP], color=INK, lw=1.2)
ax.text(np.pi/2 - 0.03, TOP - 0.02, 'boundary', rotation=90, ha='right', va='top', fontsize=7, color=INK)
th = np.linspace(0, np.pi/2, 50)
ax.plot(th, th, ls='--', color=GREY, lw=0.8); ax.plot(th[:45], -th[:45], ls='--', color=GREY, lw=0.8)
ax.text(1.0, 0.86, 'horizon', color=GREY, fontsize=7, rotation=38, ha='left')
ax.plot([LEFT, np.pi/2], [np.pi/2]*2, ls=':', color=GREY, lw=0.9)
ax.text(LEFT + 0.03, np.pi/2 + 0.05, r'$\Phi=0$', color=GREY, fontsize=7.5)
s = np.linspace(-0.75, np.pi/2 + 0.95, 500); x, t = clock(s)
ax.plot(x, t, color=INK, lw=1.6, zorder=4); xc0, tc0 = clock(-0.55); ax.text(xc0 - 0.05, tc0, 'clock', fontsize=7.5, va='center', ha='right')
YH = np.log(np.cos(ts))
labs = [(YH - 1, r'$Y_H-1$'), (YH, r'$Y_H$'), (0.0, r'$0$'), (-0.9*YH, r'$-0.9\,Y_H$')]
for y, nm in labs:
    Tb = gd(y); xc, tc = clock(np.pi/2 + tau(y))
    ax.plot([np.pi/2, xc], [Tb, tc], color=BLUE, lw=1.0, zorder=3)
    ax.plot(xc, tc, 'o', ms=4.5, mfc=BLUE, mec='white', mew=0.8, zorder=5)
    ax.text(np.pi/2 + 0.05, Tb, nm, va='center', fontsize=7.5)
    c = tc - xc                                                      # outgoing ray: T - theta = c
    xe = min(np.pi/2, TOP - c)
    ax.plot([xc, xe], [tc, xe + c], color=ORANGE, lw=1.0, zorder=2)
ax.plot([-0.92, -0.77], [-0.80, -0.80], color=BLUE, lw=1.0); ax.text(-0.73, -0.80, 'ingoing signal', va='center', fontsize=7)
ax.plot([-0.92, -0.77], [-0.95, -0.95], color=ORANGE, lw=1.0); ax.text(-0.73, -0.95, 'outgoing ray', va='center', fontsize=7)
ax.set_xlim(LEFT, np.pi/2 + 0.55); ax.set_ylim(BOT, TOP)
ax.set_xlabel(r'$\theta$'); ax.set_ylabel(r'$T$')
ax.set_xticks([-np.pi/4, 0, np.pi/4, np.pi/2]); ax.set_xticklabels([r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$'])
ax.set_yticks([-np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4]); ax.set_yticklabels([r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$'])
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig('figures/fig_geometry.pdf'); pass
