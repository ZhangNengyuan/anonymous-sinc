import numpy as np
import matplotlib.pyplot as plt
import os

# --- Data files ---
# For convenience, file names are listed here.
# Please ensure these files are located in the same directory as this script.
cwnd_file1 = 'n0-job1-cwnd-p2p-auto.txt'
cwnd_file2 = 'm0-job2-cwnd-p2p-auto.txt'
queue_file = 'n5-queueSize-p2p-auto.txt'

# Check whether required data files exist
required_files = [cwnd_file1, cwnd_file2, queue_file]
for f in required_files:
    if not os.path.exists(f):
        print(f"Error: data file '{f}' not found. Please place the required files in the same directory.")
        # Create an empty file to prevent runtime failure
        open(f, 'a').close()

# Time offset in microseconds; the simulation starts at 1 second
time_offset = 1_000_000


def read_cwnd_data(filename):
    """Read congestion window (CWND) data from file."""
    time = []
    cwnd = []

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.array([0]), np.array([0])  # Default values for missing or empty files

    with open(filename, 'r') as f:
        for line in f:
            try:
                t, c_size = map(float, line.strip().split())
                time.append(t - time_offset)
                cwnd.append(c_size)
            except ValueError:
                print(f"Warning: skipping invalid line in '{filename}': {line.strip()}")

    if not time:  # File is empty or contains only invalid lines
        return np.array([0]), np.array([0])

    return np.array(time), np.array(cwnd)


def read_queue_data(filename):
    """Read queue size data from file."""
    time = []
    queue_size = []

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.array([0]), np.array([0])  # Default values

    with open(filename, 'r') as f:
        for line in f:
            try:
                t, q_size = map(float, line.strip().split())
                time.append(t - time_offset)
                queue_size.append(q_size)
            except ValueError:
                print(f"Warning: skipping invalid line in '{filename}': {line.strip()}")

    if not time:
        return np.array([0]), np.array([0])

    return np.array(time), np.array(queue_size)


# --- Plot configuration ---
print("Generating figure...")

plt.style.use('seaborn-v0_8-whitegrid')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

fig, ax1 = plt.subplots(1, 1, figsize=(12.5, 7))

# --- Load and plot data ---
time_n0, cwnd_n0 = read_cwnd_data(cwnd_file1)
time_m0, cwnd_m0 = read_cwnd_data(cwnd_file2)
time_q, queue_size = read_queue_data(queue_file)

ax1.plot(time_n0, cwnd_n0, label='Job1 (CWND)', linewidth=1.5, color='#FF0000')
ax1.plot(time_m0, cwnd_m0, label='Job2 (CWND)', linewidth=1.5, dashes=(10, 5), color='#0000FF')
ax1.plot(time_q, queue_size, label='Queue Size', linewidth=2, color='#00AA00')

# --- Axis settings ---
ax1.set_xlabel('Time (us)', fontsize=35)
ax1.set_ylabel('CWND (Pkts)', fontsize=35)

ax1.grid(False)
ax1.tick_params(axis='both', labelsize=30)
ax1.grid(which='major', color='#D3D3D3', linestyle='--', linewidth=0.8)

# Axis limits
max_time = max(np.max(time_n0), np.max(time_m0), 10000)
ax1.set_xlim(0, max_time)
ax1.set_ylim(bottom=0)

# Axis border style
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# --- Save figure ---
# We do not use plt.tight_layout() or plt.subplots_adjust().
# Using bbox_inches='tight' ensures the entire figure is properly captured.
output_filename = 'cwnd_queue_comparison_fanin48.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure successfully saved as '{output_filename}'.")
