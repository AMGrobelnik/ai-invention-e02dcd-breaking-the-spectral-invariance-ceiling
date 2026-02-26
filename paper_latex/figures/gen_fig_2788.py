import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# Figure setup: 3:1 aspect ratio, high resolution
fig_width = 18
fig_height = 6
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans']

# Layout parameters
left_margin = 0.01
method_col_width = 0.20  # width for method names
data_col_width = (1.0 - left_margin - method_col_width - 0.01) / n_cols

top_margin = 0.92
header_height = 0.10
row_height = 0.095

# Colors
header_bg = '#2c3e50'  # dark blue-gray
header_text_color = 'white'
alt_row_colors = ['#ffffff', '#f2f2f2']
cfi_col_color = '#fff9c4'  # subtle yellow for CFI column
cfi_header_color = '#f0e68c'  # slightly stronger yellow for CFI header
bold_bg = '#e8f5e9'  # very light green for bold total cells
grid_color = '#bdbdbd'

font_size_header = 11
font_size_data = 11
font_size_method = 10.5

# CFI column index (0-based in data columns)
cfi_idx = 6
total_idx = 10

# Draw function
def draw_cell(x, y, w, h, text, bg_color='white', text_color='black',
              fontsize=11, fontweight='normal', ha='center', fontfamily='sans-serif'):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg_color, edgecolor=grid_color, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            fontfamily=fontfamily)

# Title
ax.text(0.5, 0.98, 'Equivariant Method Discrimination Counts by Graph Category',
        ha='center', va='top', fontsize=14, fontweight='bold', fontfamily='sans-serif',
        color='#2c3e50')

# Subtitle with threshold info
ax.text(0.5, 0.945, 'Number of distinguished non-isomorphic graph pairs per category (threshold: 1e-6)',
        ha='center', va='top', fontsize=9.5, fontfamily='sans-serif', color='#666666')

# Draw header row
y_header = top_margin - header_height

# Method header cell
draw_cell(left_margin, y_header, method_col_width, header_height,
          'Method', bg_color=header_bg, text_color=header_text_color,
          fontsize=font_size_header, fontweight='bold')

# Data column headers
for j, hdr in enumerate(col_headers):
    x = left_margin + method_col_width + j * data_col_width
    bg = cfi_header_color if j == cfi_idx else header_bg
    tc = '#333333' if j == cfi_idx else header_text_color
    fw = 'bold'
    draw_cell(x, y_header, data_col_width, header_height,
              hdr, bg_color=bg, text_color=tc,
              fontsize=font_size_header - 0.5, fontweight=fw)

# Draw data rows
for i, (method, row_data) in enumerate(zip(methods, data)):
    y = y_header - (i + 1) * row_height
    row_bg = alt_row_colors[i % 2]

    # Method name cell
    draw_cell(left_margin, y, method_col_width, row_height,
              method, bg_color=row_bg, text_color='#333333',
              fontsize=font_size_method, fontweight='normal', ha='left',
              fontfamily='sans-serif')

    # Data cells
    for j, val in enumerate(row_data):
        x = left_margin + method_col_width + j * data_col_width

        # Determine background color
        if j == cfi_idx:
            cell_bg = cfi_col_color
        else:
            cell_bg = row_bg

        # Determine if this total should be bold (346)
        fw = 'normal'
        tc = '#333333'
        if j == total_idx and val == 346:
            fw = 'bold'
            tc = '#1a237e'  # dark blue for emphasis
            cell_bg = '#e8eaf6'  # light indigo background

        # Format value
        val_str = str(val)

        draw_cell(x, y, data_col_width, row_height,
                  val_str, bg_color=cell_bg, text_color=tc,
                  fontsize=font_size_data, fontweight=fw,
                  fontfamily='sans-serif')

# Save
plt.tight_layout(pad=0.5)
output_path = '/workspace/runs/run__20260226_110200/4_gen_paper_repo/figures/fig_2788_all/fig_2788_v0_it1.png'
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved to {output_path}")
