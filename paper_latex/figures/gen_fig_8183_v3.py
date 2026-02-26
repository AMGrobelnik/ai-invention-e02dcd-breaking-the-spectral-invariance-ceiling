import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

blue   = '#2171b5'
orange = '#e6550d'
red    = '#cb181d'

# ── 11 explicitly described data points ───────────────────────────────
# Slight x-jitter in the 345-346 cluster for readability
# (x, y, color, marker, size)
data = [
    (344.0, 0.0908,  blue,   'o', 110),   # RWPE + GINEConv  (labeled)
    (346.0, 0.1026,  orange, 'o', 110),   # nRWPE-diag + GINEConv
    (348.0, 0.1716,  orange, '^', 110),   # nRWPE-offdiag + GINEConv
    (343.5, 0.1707,  blue,   's', 110),   # RWPE + GIN_v2
    (346.5, 0.1825,  orange, 's', 110),   # nRWPE-diag + GIN_v2
    (343.0, 0.1737,  blue,   'D', 110),   # RWPE + GPS
    (347.5, 0.2959,  orange, 'D', 110),   # nRWPE-offdiag + GPS
    (345.5, 0.3055,  orange, 'p', 110),   # nRWPE-combined + GPS
    (525.0, 0.3354,  red,    '*', 250),   # KW-PE + GIN_v1  (labeled)
    (344.5, 0.1845,  blue,   's', 70),    # RWPE + GIN_v1 (smaller)
    (347.0, 0.3198,  orange, 's', 70),    # nRWPE-multi + GIN_v2 (smaller)
]

# ── FIGURE ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

xs_all, ys_all = [], []
for (x, y, c, m, s) in data:
    ax.scatter(x, y, c=c, marker=m, s=s,
               edgecolors='white', linewidths=0.7, zorder=5)
    xs_all.append(x)
    ys_all.append(y)

# ── Trend line (dashed gray) ──────────────────────────────────────────
xs_arr = np.array(xs_all)
ys_arr = np.array(ys_all)
z = np.polyfit(xs_arr, ys_arr, 1)
p_line = np.poly1d(z)
x_line = np.linspace(336, 535, 200)
ax.plot(x_line, p_line(x_line), '--', color='#aaaaaa', linewidth=1.3, zorder=2)

# ── Annotations ───────────────────────────────────────────────────────
# RWPE + GINEConv — bottom-left best performer
ax.annotate('RWPE + GINEConv',
            xy=(344, 0.0908), xytext=(395, 0.065),
            fontsize=10, fontweight='semibold', color=blue,
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0,
                            connectionstyle='arc3,rad=-0.1'),
            ha='left', va='center', zorder=10)

# KW-PE — top-right outlier
ax.annotate('Non-equivariant KW-PE',
            xy=(525, 0.3354), xytext=(425, 0.360),
            fontsize=10, fontweight='semibold', color=red,
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0,
                            connectionstyle='arc3,rad=0.15'),
            ha='center', va='bottom', zorder=10)

# ── Spearman annotation ──────────────────────────────────────────────
ax.text(0.98, 0.03,
        r'Spearman $\rho$ = 0.42, $p$ = 0.10 (n.s.)',
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#bbbbbb', alpha=0.9))

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlabel('Pair Discrimination Count (out of 525)', fontsize=13, labelpad=10)
ax.set_ylabel('ZINC-12k Test MAE (lower is better)', fontsize=13, labelpad=10)
ax.set_xlim(335, 540)
ax.set_ylim(0.04, 0.39)
ax.set_xticks([340, 360, 380, 400, 420, 440, 460, 480, 500, 520])
ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

ax.grid(True, linestyle=':', alpha=0.3, color='#cccccc')
ax.set_axisbelow(True)

# ── LEGEND ────────────────────────────────────────────────────────────
# Architecture shapes
arch_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#777777',
           markeredgecolor='#444444', markersize=9, linestyle='None', label='GINEConv'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#777777',
           markeredgecolor='#444444', markersize=9, linestyle='None', label='GIN'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#777777',
           markeredgecolor='#444444', markersize=9, linestyle='None', label='GPS'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#777777',
           markeredgecolor='#444444', markersize=12, linestyle='None', label='GIN (KW-PE)'),
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

# Place Architecture legend in the empty space (upper-left area)
leg1 = ax.legend(handles=arch_handles, title='Architecture',
                 loc='upper left', fontsize=9.5, title_fontsize=10.5,
                 framealpha=0.95, edgecolor='#cccccc',
                 handletextpad=0.6,
                 bbox_to_anchor=(0.02, 0.98))
ax.add_artist(leg1)

# Method Type legend below Architecture
leg2 = ax.legend(handles=color_handles, title='Method Type',
                 loc='upper left', fontsize=9.5, title_fontsize=10.5,
                 framealpha=0.95, edgecolor='#cccccc',
                 handletextpad=0.6,
                 bbox_to_anchor=(0.02, 0.70))

# ── Save ──────────────────────────────────────────────────────────────
fig.tight_layout()
out = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_8183_all/fig_8183_v0_it3.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
