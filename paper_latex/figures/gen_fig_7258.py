#!/usr/bin/env python3
"""Generate ZINC-12k Molecular Regression Performance grouped bar chart - v4 final."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use sans-serif fonts
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans'],
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Colors
C_BLUE = '#3274A1'    # RWPE variants
C_ORANGE = '#E8803A'  # nRWPE variants
C_RED = '#C44E52'     # KW-PE
C_GRAY = '#8C8C8C'    # No PE
C_GREEN = '#55A868'   # LapPE

# Data definition
groups = [
    ("GIN_v1", [
        ("No PE",   0.2849, 0, 0, C_GRAY,  False),
        ("RWPE",    0.1845, 0, 0, C_BLUE,  True),
        ("LapPE",   0.2394, 0, 0, C_GREEN, False),
        ("KW-PE",   0.3354, 0, 0, C_RED,   False),
    ]),
    ("GIN_v2", [
        ("RWPE",          0.1707, 0.0025, 0.0025, C_BLUE,   True),
        ("nRWPE-\ndiag",  0.1825, 0.0037, 0.0037, C_ORANGE, False),
        ("nRWPE-\nmulti", 0.3198, 0.1241, 0.1241, C_ORANGE, False),
    ]),
    ("GINEConv", [
        ("RWPE-16",           0.0908, 0.0024, 0.0024, C_BLUE,   True),
        ("nRWPE-\ndiag-tanh", 0.1026, 0.0017, 0.0017, C_ORANGE, False),
        ("nRWPE-\noffdiag",   0.1716, 0.0035, 0.0035, C_ORANGE, False),
        ("nRWPE-\ncombined",  0.1716, 0.0031, 0.0031, C_ORANGE, False),
    ]),
    ("GPS", [
        ("RWPE-8",             0.1737, 0.0004, 0.0004, C_BLUE,   True),
        ("nRWPE-\noffdiag-8",  0.2959, 0.0026, 0.0026, C_ORANGE, False),
        ("nRWPE-\ncombined-8", 0.3055, 0.0092, 0.0092, C_ORANGE, False),
        ("No PE",              0.3040, 0.0015, 0.0015, C_GRAY,   False),
    ]),
]

fig, ax = plt.subplots(figsize=(18, 8.5))

bar_width = 0.75
bar_spacing = 0.28
group_gap = 2.5

x_positions = []
x_labels = []
group_center_positions = []
group_labels_list = []
all_group_ranges = []

current_x = 0
for g_idx, (g_label, bars) in enumerate(groups):
    positions_in_group = []
    start_x = current_x
    for b_idx, (b_label, val, err_lo, err_hi, color, is_best) in enumerate(bars):
        x = current_x + b_idx * (bar_width + bar_spacing)
        positions_in_group.append(x)

        # Draw bar
        err = [[err_lo], [err_hi]] if (err_lo > 0 or err_hi > 0) else None
        ax.bar(x, val, width=bar_width, color=color, edgecolor='white',
               linewidth=0.6, zorder=3,
               yerr=err, capsize=4,
               error_kw=dict(elinewidth=1.3, capthick=1.1, color='#333333', zorder=4))

        # Value annotation above bar
        top = val + err_hi if err_hi > 0 else val
        ax.text(x, top + 0.007, f'{val:.4f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='normal', color='#333333', rotation=0)

        # Star for best in group
        if is_best:
            ax.text(x, top + 0.026, '★', ha='center', va='bottom',
                    fontsize=16, color='#D4AF37', zorder=5)

        x_positions.append(x)
        x_labels.append(b_label)

    end_x = positions_in_group[-1]
    group_center = np.mean(positions_in_group)
    group_center_positions.append(group_center)
    group_labels_list.append(g_label)
    all_group_ranges.append((start_x - bar_width/2, end_x + bar_width/2))

    current_x = end_x + bar_width + group_gap

# X-axis
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, ha='center', fontsize=9.5, linespacing=0.9)

# Group labels below x-axis
for gc, gl in zip(group_center_positions, group_labels_list):
    ax.text(gc, -0.075, gl, ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.get_xaxis_transform())

# Vertical separators between groups
for i in range(len(all_group_ranges) - 1):
    mid_x = (all_group_ranges[i][1] + all_group_ranges[i+1][0]) / 2
    ax.axvline(x=mid_x, color='#CCCCCC', linestyle=':', linewidth=0.8, zorder=1)

# Y-axis
ax.set_ylim(0, 0.48)
ax.set_yticks(np.arange(0, 0.50, 0.05))
ax.set_ylabel('Test MAE (lower is better)', fontsize=16, labelpad=10)

# Dashed horizontal line for best overall
ax.axhline(y=0.0908, color='#666666', linestyle='--', linewidth=1.3, zorder=2, alpha=0.7)

# "Best overall" annotation - place in the upper-right region with arrow
# Position text above GPS group area where there's more whitespace
gps_center = group_center_positions[3]
ax.annotate('Best overall: GINEConv + RWPE-16 (0.0908)',
            xy=(gps_center, 0.0908),
            xytext=(gps_center, 0.44),
            fontsize=11, fontstyle='italic', color='#555555', ha='center',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0, shrinkA=0, shrinkB=3),
            zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.9))

# Title
ax.set_title('Downstream ZINC-12k Molecular Regression Performance',
             fontsize=19, fontweight='bold', pad=16)

# Legend
legend_handles = [
    mpatches.Patch(color=C_BLUE, label='RWPE variants'),
    mpatches.Patch(color=C_ORANGE, label='nRWPE variants'),
    mpatches.Patch(color=C_GREEN, label='LapPE'),
    mpatches.Patch(color=C_RED, label='KW-PE (EDMD)'),
    mpatches.Patch(color=C_GRAY, label='No PE'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#D4AF37',
               markersize=16, label='Best in group'),
]
ax.legend(handles=legend_handles, loc='upper left', framealpha=0.95,
          edgecolor='#CCCCCC', ncol=3, fontsize=11, handlelength=1.5,
          borderpad=0.8, columnspacing=1.2, bbox_to_anchor=(0.0, 1.0))

# Grid
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#999999', zorder=0)
ax.set_axisbelow(True)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Margins
ax.set_xlim(x_positions[0] - bar_width - 0.3, x_positions[-1] + bar_width + 0.3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)

# Save
output_path = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_7258_all/fig_7258_v0_it4.png'
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to {output_path}')
