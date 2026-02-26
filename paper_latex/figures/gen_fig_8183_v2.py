import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D

# ── NeurIPS style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 9.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

blue   = '#1f77b4'
orange = '#ff7f0e'
red    = '#d62728'

# ── ONLY the 11 explicitly described data points ─────────────────────
# (x, y, short_name, color, marker, size)
data = [
    (345, 0.0908,  'RWPE + GINEConv',         blue,   'o', 100),   # labeled
    (345, 0.1026,  'nRWPE-diag + GINEConv',   orange, 'o', 100),
    (346, 0.1716,  'nRWPE-offdiag + GINEConv', orange, '^', 100),  # triangle per spec
    (345, 0.1707,  'RWPE + GIN_v2',           blue,   's', 100),
    (345, 0.1825,  'nRWPE-diag + GIN_v2',     orange, 's', 100),
    (345, 0.1737,  'RWPE + GPS',              blue,   'D', 100),
    (346, 0.2959,  'nRWPE-offdiag + GPS',     orange, 'D', 100),
    (346, 0.3055,  'nRWPE-combined + GPS',    orange, 'p', 100),   # pentagon
    (525, 0.3354,  'KW-PE + GIN_v1',          red,    '*', 220),   # labeled
    (345, 0.1845,  'RWPE + GIN_v1',           blue,   's', 65),    # smaller square
    (345, 0.3198,  'nRWPE-multi + GIN_v2',    orange, 's', 65),    # smaller square
]

# ── FIGURE ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

xs_all, ys_all = [], []
for (x, y, name, c, m, s) in data:
    ax.scatter(x, y, c=c, marker=m, s=s, edgecolors='white', linewidths=0.6, zorder=5)
    xs_all.append(x)
    ys_all.append(y)

# ── Trend line (dashed gray) ──────────────────────────────────────────
xs_arr = np.array(xs_all)
ys_arr = np.array(ys_all)
z = np.polyfit(xs_arr, ys_arr, 1)
p_line = np.poly1d(z)
x_line = np.linspace(338, 532, 100)
ax.plot(x_line, p_line(x_line), '--', color='#999999', linewidth=1.3, alpha=0.7, zorder=2)

# ── Annotations ───────────────────────────────────────────────────────
# RWPE + GINEConv at (345, 0.0908)  – bottom-left best performer
ax.annotate('RWPE + GINEConv',
            xy=(345, 0.0908), xytext=(390, 0.060),
            fontsize=9.5, fontweight='semibold', color=blue,
            arrowprops=dict(arrowstyle='->', color='#777777', lw=0.9),
            ha='left', va='center', zorder=10)

# KW-PE + GIN_v1 at (525, 0.3354) – top-right outlier
ax.annotate('Non-equivariant KW-PE',
            xy=(525, 0.3354), xytext=(430, 0.355),
            fontsize=9.5, fontweight='semibold', color=red,
            arrowprops=dict(arrowstyle='->', color='#777777', lw=0.9),
            ha='center', va='bottom', zorder=10)

# ── Spearman annotation box ──────────────────────────────────────────
ax.text(0.98, 0.03,
        r'Spearman $\rho$ = 0.42, $p$ = 0.10 (n.s.)',
        transform=ax.transAxes, fontsize=9.5, ha='right', va='bottom',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f5',
                  edgecolor='#aaaaaa', alpha=0.9))

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlabel('Pair Discrimination Count (out of 525)', fontsize=13, labelpad=8)
ax.set_ylabel('ZINC-12k Test MAE (lower is better)', fontsize=13, labelpad=8)
ax.set_xlim(335, 540)
ax.set_ylim(0.04, 0.38)
ax.set_xticks([340, 360, 380, 400, 420, 440, 460, 480, 500, 520])
ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

# Light grid
ax.grid(True, linestyle=':', alpha=0.3, color='#bbbbbb')
ax.set_axisbelow(True)

# ── LEGEND ────────────────────────────────────────────────────────────
# Architecture shapes
arch_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
           markeredgecolor='#555555', markersize=9, linestyle='None', label='GINEConv'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#888888',
           markeredgecolor='#555555', markersize=9, linestyle='None', label='GIN'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#888888',
           markeredgecolor='#555555', markersize=9, linestyle='None', label='GPS'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#888888',
           markeredgecolor='#555555', markersize=11, linestyle='None', label='GIN (KW-PE)'),
]

# Method type colors
color_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=blue,
           markeredgecolor='white', markersize=9, linestyle='None', label='RWPE'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=orange,
           markeredgecolor='white', markersize=9, linestyle='None', label='Equiv. nRWPE'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=red,
           markeredgecolor='white', markersize=9, linestyle='None', label='Non-equiv. KW-PE'),
]

leg1 = ax.legend(handles=arch_handles, title='Architecture',
                 loc='upper left', fontsize=9, title_fontsize=10,
                 framealpha=0.95, edgecolor='#cccccc',
                 bbox_to_anchor=(0.01, 0.99))
ax.add_artist(leg1)

leg2 = ax.legend(handles=color_handles, title='Method Type',
                 loc='upper left', fontsize=9, title_fontsize=10,
                 framealpha=0.95, edgecolor='#cccccc',
                 bbox_to_anchor=(0.01, 0.72))

# ── Save ──────────────────────────────────────────────────────────────
fig.tight_layout()
out = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_8183_all/fig_8183_v0_it2.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
