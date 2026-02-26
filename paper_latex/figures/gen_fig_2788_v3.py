import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
methods = [
    'RWPE-diag-K20',
    'Linear-walk-diag',
    'nRWPE-diag-tanh-T20',
    'nRWPE-diag-tanh-T50',
    'nRWPE-offdiag-tanh-T20',
    'nRWPE-diag-softplus',
    'nRWPE-diag-ReLU',
]

col_headers = [
    'Cospectral\n(64)', 'CSL\n(59)', 'Str.Reg.\n(2)', 'Basic\n(60)',
    'Regular\n(50)', 'Extension\n(100)', 'CFI\n(100)', '4-Vertex\n(20)',
    'Dist-Reg\n(20)', 'Str-Reg\n(50)', 'Total\n(525)'
]

data = [
    [64, 59, 0, 60, 50, 100, 12, 0, 0, 0, 345],
    [64, 59, 0, 60, 50, 100, 11, 0, 0, 0, 344],
    [64, 59, 0, 60, 50, 100, 12, 0, 0, 0, 345],
    [64, 59, 0, 60, 50, 100, 13, 0, 0, 0, 346],
    [64, 59, 0, 60, 50, 100, 13, 0, 0, 0, 346],
    [64, 59, 0, 60, 50, 100,  8, 0, 0, 0, 341],
    [64, 59, 0, 60, 50, 100, 11, 0, 0, 0, 344],
]

n_rows = len(methods)
n_cols = len(col_headers)

# Figure: 3:1 wide aspect, high res
fig_width = 16
fig_height = 5.8
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=250)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']

# Layout
left_margin = 0.005
method_col_width = 0.19
remaining = 1.0 - left_margin - method_col_width - 0.005
data_col_width = remaining / n_cols

top_start = 0.895
header_height = 0.092
row_height = 0.092

# Colors
header_bg = '#37474f'
header_text_color = 'white'
alt_colors = ['#ffffff', '#f5f5f5']
cfi_bg = '#fff8e1'
cfi_header_bg = '#ffe082'
grid_color = '#cfd8dc'
total_bold_bg = '#e3f2fd'

font_header = 10.5
font_data = 10.5
font_method = 10

cfi_idx = 6
total_idx = 10

def draw_cell(x, y, w, h, text, bg='white', tc='#212121', fs=10.5, fw='normal', ff='sans-serif', ha='center'):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=grid_color, linewidth=0.7, clip_on=False)
    ax.add_patch(rect)
    if ha == 'left':
        ax.text(x + 0.008, y + h/2, text, ha='left', va='center',
                fontsize=fs, fontweight=fw, color=tc, fontfamily=ff)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, fontfamily=ff)

# Title
ax.text(0.5, 0.985, 'Equivariant Method Discrimination Counts by Graph Category',
        ha='center', va='top', fontsize=14.5, fontweight='bold', fontfamily='sans-serif',
        color='#263238')

ax.text(0.5, 0.935, 'Number of distinguished non-isomorphic graph pairs per category (threshold: 1e\u22126)',
        ha='center', va='top', fontsize=9.5, fontfamily='sans-serif', color='#757575')

# Header row
y_h = top_start - header_height

# Top border
table_left = left_margin
table_right = left_margin + method_col_width + n_cols * data_col_width
ax.plot([table_left, table_right], [y_h + header_height, y_h + header_height],
        color='#455a64', linewidth=2.0, clip_on=False)

# Method header
draw_cell(left_margin, y_h, method_col_width, header_height,
          'Method', bg=header_bg, tc=header_text_color, fs=font_header, fw='bold')

# Column headers
for j, hdr in enumerate(col_headers):
    x = left_margin + method_col_width + j * data_col_width
    if j == cfi_idx:
        bg = cfi_header_bg
        tc = '#333333'
    elif j == total_idx:
        bg = '#263238'
        tc = 'white'
    else:
        bg = header_bg
        tc = header_text_color
    draw_cell(x, y_h, data_col_width, header_height, hdr, bg=bg, tc=tc, fs=font_header - 0.5, fw='bold')

# Separator line below header
ax.plot([table_left, table_right], [y_h, y_h],
        color='#455a64', linewidth=1.5, clip_on=False)

# Data rows
for i, (method, row_data) in enumerate(zip(methods, data)):
    y = y_h - (i + 1) * row_height
    row_bg = alt_colors[i % 2]

    # Method cell
    draw_cell(left_margin, y, method_col_width, row_height,
              method, bg=row_bg, tc='#37474f', fs=font_method, fw='normal', ha='left')

    for j, val in enumerate(row_data):
        x = left_margin + method_col_width + j * data_col_width

        # Background
        if j == cfi_idx:
            cell_bg = cfi_bg
        elif j == total_idx and val == 346:
            cell_bg = total_bold_bg
        else:
            cell_bg = row_bg

        # Text styling
        fw = 'normal'
        tc = '#424242'
        fs = font_data
        if j == total_idx and val == 346:
            fw = 'bold'
            tc = '#0d47a1'
            fs = font_data + 0.5
        elif j == total_idx:
            tc = '#37474f'
            fw = 'medium'

        draw_cell(x, y, data_col_width, row_height, str(val),
                  bg=cell_bg, tc=tc, fs=fs, fw=fw)

# Bottom border
y_bottom = y_h - n_rows * row_height
ax.plot([table_left, table_right], [y_bottom, y_bottom],
        color='#455a64', linewidth=2.0, clip_on=False)

# Save
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
output_path = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_2788_all/fig_2788_v0_it3.png'
fig.savefig(output_path, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.15)
plt.close()
print(f"Saved to {output_path}")
