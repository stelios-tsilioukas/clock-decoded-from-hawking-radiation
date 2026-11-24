"""Fig. 2: decoder correction k1*(decoded - one-sided) vs physical stop label Y, and its two decoupling pieces.
k1 = 20, tau_s' = pi/4, m = 2, omega = 0.2, h = 1, eps' ~ 0.05, Delta_e = 1, eps_e = 0.0125 (deep post-Page).
Data: independent re-derivation run (rederivation/radiating-decoder/analysis.md, A8, k1 = 20), which reproduces Step 4e."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'axes.linewidth': 0.6})
BLUE, ORANGE, INK, GREY = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8a86'
rows = [(-3.0, -0.22182, -0.03531, -0.18652), (-2.5, -0.14403, -0.04270, -0.10133), (-2.0, -0.08466, -0.05325, -0.03141),
        (-1.5, -0.04343, -0.06799, 0.02455), (-1.3466, -0.03416, -0.07328, 0.03911), (-1.0, -0.01837, -0.08440, 0.06603),
        (-0.75, -0.01080, -0.08656, 0.07577), (-0.5, -0.00571, -0.07500, 0.06929), (-0.3466, -0.00352, -0.05830, 0.05479),
        (-0.25, -0.00239, -0.04423, 0.04184), (-0.1, -0.00090, -0.01850, 0.01760)]
Y = np.array([r[0] for r in rows]); tot = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows]); G = np.array([r[3] for r in rows])
Y = np.r_[Y, 0.0, -Y[::-1]]; tot = np.r_[tot, 0.0, -tot[::-1]]; D = np.r_[D, 0.0, -D[::-1]]; G = np.r_[G, 0.0, -G[::-1]]   # odd in Y (exact)
YH = np.log(np.cos(np.pi/4))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.axhline(0, color=GREY, lw=0.5)
for x0 in (YH, -0.9*YH):
    ax.axvline(x0, color=GREY, lw=0.6, ls=':')

ax.text(YH - 0.05, -0.215, r'$Y_H$', ha='right', fontsize=7, color=GREY)
ax.axvspan(YH, -0.9*YH, color='#ececea', zorder=0)
ax.plot(Y, D, color=BLUE, lw=1.4, label='brane-side ($D$) decoupling')
ax.plot(Y, G, color=ORANGE, lw=1.4, label='gap decoupling')
ax.plot(Y, tot, color=INK, lw=1.8, label='total decoder correction')
yy = np.linspace(-3, 3, 200); ax.plot(yy, 0.0093*yy + 0.0091*yy**3 - 0.0002*yy**5, color=INK, lw=0.7, ls='--')
ax.set_xlim(-3.1, 3.1); ax.set_ylim(-0.24, 0.24)
ax.set_xlabel(r'stop label $Y$'); ax.set_ylabel(r'$k_1\,(T_{\rm dec}-T)$')
ax.legend(frameon=False, fontsize=6.8, loc='upper left', handlelength=1.6)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig('figures/fig_decoder.pdf'); pass
