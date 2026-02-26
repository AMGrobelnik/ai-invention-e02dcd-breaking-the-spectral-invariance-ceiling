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

# Figure setup: 3:1 wide, high resolution
fig_width = 16
fig_height = 5.5
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=250)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']

# Layout
left_margin = 0.005
method_col_width = 0.185
remaining = 1.0 - left_margin - method_col_width - 0.005
data_col_width = remaining / n_cols

top_start = 0.90
header_height = 0.09
row_height = 0.088

# Colors
header_bg = '#37474f'       # blue-gray 800
header_text_color = 'white'
alt_colors = ['#ffffff', '#f5f5f5']
cfi_bg = '#fff8e1'          # amber 50 - subtle yellow
cfi_header_bg = '#ffe082'   # amber 200
grid_color = '#cfd8dc'      # blue-gray 100
total_bold_bg = '#e3f2fd'   # blue 50

font_header = 10.5
font_data = 10.5
font_method = 10

cfi_idx = 6
total_idx = 10

def draw_cell(x, y, w, h, text, bg='white', tc='#212121', fs=10.5, fw='normal', ff='sans-serif'):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=grid_color, linewidth=0.7, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, fontfamily=ff)

def draw_cell_left(x, y, w, h, text, bg='white', tc='#212121', fs=10, fw='normal', ff='sans-serif'):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=grid_color, linewidth=0.7, clip_on=False)
    ax.add_patch(rect)
    ax.text(x + 0.008, y + h/2, text, ha='left', va='center',
            fontsize=fs, fontweight=fw, color=tc, fontfamily=ff)

# Title
ax.text(0.5, 0.98, 'Equivariant Method Discrimination Counts by Graph Category',
        ha='center', va='top', fontsize=14, fontweight='bold', fontfamily='sans-serif',
        color='#263238')

ax.text(0.5, 0.935, 'Number of distinguished non-isomorphic graph pairs per category (threshold: 1e\u22126)',
        ha='center', va='top', fontsize=9, fontfamily='sans-serif', color='#757575')

# Header row
y_h = top_start - header_height

# Method header
draw_cell(left_margin, y_h, method_col_width, header_height,
          'Method', bg=header_bg, tc=header_text_color, fs=font_header, fw='bold')

# Column headers
for j, hdr in enumerate(col_headers):
    x = left_margin + method_col_width + j * data_col_width
    if j == cfi_idx:
        bg = cfi_header_bg
        tc = '#424242'
    elif j == total_idx:
        bg = '#263238'  # darker for total
        tc = 'white'
    else:
        bg = header_bg
        tc = header_text_color
    draw_cell(x, y_h, data_col_width, header_height, hdr, bg=bg, tc=tc, fs=font_header - 0.5, fw='bold')

# Data rows
for i, (method, row_data) in enumerate(zip(methods, data)):
    y = y_h - (i + 1) * row_height
    row_bg = alt_colors[i % 2]

    # Method cell (left-aligned)
    draw_cell_left(left_margin, y, method_col_width, row_height,
                   method, bg=row_bg, tc='#37474f', fs=font_method, fw='normal')

    for j, val in enumerate(row_data):
        x = left_margin + method_col_width + j * data_col_width

        # Cell background
        if j == cfi_idx:
            cell_bg = cfi_bg
        elif j == total_idx and val == 346:
            cell_bg = total_bold_bg
        else:
            cell_bg = row_bg

        # Text style
        fw = 'normal'
        tc = '#424242'
        if j == total_idx and val == 346:
            fw = 'bold'
            tc = '#0d47a1'  # strong blue
        elif j == total_idx:
            tc = '#37474f'

        draw_cell(x, y, data_col_width, row_height, str(val),
                  bg=cell_bg, tc=tc, fs=font_data, fw=fw)

# Bottom border - thicker line at the bottom of the table
y_bottom = y_h - n_rows * row_height
ax.plot([left_margin, left_margin + method_col_width + n_cols * data_col_width],
        [y_bottom, y_bottom], color='#607d8b', linewidth=1.5, clip_on=False)

# Top border of header
ax.plot([left_margin, left_margin + method_col_width + n_cols * data_col_width],
        [y_h + header_height, y_h + header_height], color='#607d8b', linewidth=1.5, clip_on=False)

# Save
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
output_path = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_2788_all/fig_2788_v0_it2.png'
fig.savefig(output_path, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.15)
plt.close()
print(f"Saved to {output_path}")
