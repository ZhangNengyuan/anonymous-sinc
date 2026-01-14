#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a standalone legend-only figure that combines legend entries from:
  (1) CWND + Queue plot
  (2) Throughput plot

Font handling is kept EXACTLY consistent with previous figures:
- seaborn-v0_8-whitegrid
- Times New Roman (serif)
- pdf.fonttype = 42
- ps.fonttype  = 42
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Style & font settings
# (EXACTLY same as previous scripts)
# =========================
plt.style.use('seaborn-v0_8-whitegrid')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # fix minus sign rendering

# =========================
# Legend entries
# (Styles match your two figures exactly)
# =========================
handles = [
    # --- CWND / Queue figure ---
    Line2D([0], [0], color='#FF0000', lw=1.5, label='Job1 (CWND)'),
    Line2D([0], [0], color='#0000FF', lw=1.5, dashes=(10, 5), label='Job2 (CWND)'),
    Line2D([0], [0], color='#00AA00', lw=2.0, label='Queue Size'),

    # --- Throughput figure ---
    Line2D([0], [0], color='#FF7070', lw=1.5, alpha=0.6, label='Job1 (100us)'),
    Line2D([0], [0], color='#7070FF', lw=1.5, alpha=0.6, dashes=(10, 5), label='Job2 (100us)'),
    Line2D([0], [0], color='#E60000', lw=2.5, marker='^', markersize=8, label='Job1 (Avg)'),
    Line2D([0], [0], color='#0000B3', lw=2.5, dashes=(10, 5),
           marker='^', markersize=8, label='Job2 (Avg)'),
]

labels = [h.get_label() for h in handles]

# =========================
# Legend-only figure
# =========================
# Wide + short figure to force ONE ROW legend
fig = plt.figure(figsize=(24, 1.6))
ax = fig.add_subplot(111)
ax.axis('off')

ax.legend(
    handles,
    labels,
    loc='center',
    ncol=len(handles),     # force single row
    frameon=True,
    fontsize=38,           # consistent with your main figures
    columnspacing=0.3,
    handlelength=2.2,
    handletextpad=0.5,
    borderpad=0.3
)

# =========================
# Save
# =========================
out_pdf = "legend.pdf"
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.close()

print(f"Legend-only figure saved: {out_pdf}")
