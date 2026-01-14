import numpy as np
import matplotlib.pyplot as plt
import os

# --- Data files ---
throughput_file1 = 'job1-sinkBytes-p2p-auto.txt'
throughput_file2 = 'job2-sinkBytes-p2p-auto.txt'

# Check whether required data files exist
required_files = [throughput_file1, throughput_file2]
for f in required_files:
    if not os.path.exists(f):
        print(f"Error: data file '{f}' not found. Please place the required files in the same directory.")
        # Create an empty file to prevent runtime failure
        open(f, 'a').close()

# Time offset in microseconds; the simulation starts at 1 second
time_offset = 1_000_000


def read_throughput_data(filename):
    """
    Read the text file and compute throughput.

    Returns:
      - timestamps (us)
      - throughput (Gbps)

    Note:
      The original data granularity is 100 us per sample.
    """
    time = []
    throughput = []

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.array([]), np.array([])  # Default values for missing or empty files

    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    timestamp_us = int(parts[0])
                    bytes_per = int(parts[2])

                    time.append(timestamp_us - time_offset)

                    # Throughput (Gbps) = bytes * 8 / (100us) / 1e9
                    throughput_gbps = bytes_per * 8 / 100e-6 / 1e9
                    throughput.append(throughput_gbps)
            except (ValueError, IndexError):
                print(f"Warning: skipping invalid line in '{filename}': {line.strip()}")

    if not time:
        return np.array([]), np.array([])

    return np.array(time), np.array(throughput)


def calculate_custom_coarse_throughput(time, throughput):
    """
    Compute coarse-grained throughput based on pre-defined non-uniform periods.

    Period definition (based on indices of 101 points):
      [0-35], [36-51], [52-68], [69-84], [85-99]

    This function applies the same index ratios to the actual number of data points.
    """
    if len(throughput) == 0:
        return np.array([]), np.array([])

    coarse_time = []
    coarse_throughput = []

    num_points = len(throughput)

    # Boundaries based on 0..100 indexing (101 points):
    # 0, 36, 52, 69, 85, 100
    period_boundaries = [0, 36, 52, 69, 85, 100]

    # Scale boundaries to actual data length
    scaled_boundaries = [int(b / 100.0 * num_points) for b in period_boundaries]

    for i in range(len(scaled_boundaries) - 1):
        start_index = scaled_boundaries[i]

        # Ensure the last period includes all remaining points
        if i == len(scaled_boundaries) - 2:
            end_index = num_points
        else:
            end_index = scaled_boundaries[i + 1]

        if start_index >= end_index:
            continue

        time_slice = time[start_index:end_index]
        throughput_slice = throughput[start_index:end_index]

        if len(throughput_slice) > 0:
            avg_throughput = np.mean(throughput_slice)
            avg_time = np.mean(time_slice)

            coarse_time.append(avg_time)
            coarse_throughput.append(avg_throughput)

    return np.array(coarse_time), np.array(coarse_throughput)


# --- Main plotting logic ---
print("Generating figure...")

plt.style.use('seaborn-v0_8-whitegrid')

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# Keep figsize consistent with the CWND/queue figure
fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))

# Load data
time1, throughput1 = read_throughput_data(throughput_file1)
time2, throughput2 = read_throughput_data(throughput_file2)

# Compute coarse-grained data
coarse_time1, coarse_throughput1 = calculate_custom_coarse_throughput(time1, throughput1)
coarse_time2, coarse_throughput2 = calculate_custom_coarse_throughput(time2, throughput2)

# --- Plotting ---
# Larger zorder is drawn on top
if len(time1) > 0:
    ax.plot(time1, throughput1,
            label='Job1 (100us)',
            linewidth=1.5,
            color='#FF7070',
            alpha=0.6,
            zorder=1)

if len(time2) > 0:
    ax.plot(time2, throughput2,
            label='Job2 (100us)',
            linewidth=1.5,
            dashes=(10, 5),
            color='#7070FF',
            alpha=0.6,
            zorder=1)

if len(coarse_time1) > 0:
    ax.plot(coarse_time1, coarse_throughput1,
            label='Job1 (Avg)',
            linewidth=2.5,
            color='#E60000',
            zorder=2,
            marker='^',
            markersize=8)

if len(coarse_time2) > 0:
    ax.plot(coarse_time2, coarse_throughput2,
            label='Job2 (Avg)',
            linewidth=2.5,
            dashes=(10, 5),
            color='#0000B3',
            zorder=2,
            marker='^',
            markersize=8)

# --- Axis settings (aligned with the CWND figure) ---
ax.set_xlabel('Time (µs)', fontsize=35)
ax.set_ylabel('Throughput (Gbps)', fontsize=35)

ax.grid(False)
ax.grid(which='major', color='#D3D3D3', linestyle='--', linewidth=0.8)
ax.tick_params(axis='both', labelsize=30)

# Axis limits
max_time = 0
if len(time1) > 0:
    max_time = max(max_time, np.max(time1))
if len(time2) > 0:
    max_time = max(max_time, np.max(time2))

ax.set_xlim(0, max(max_time, 10000))
ax.set_ylim(bottom=0)

# Axis border style
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# --- Save figure ---
output_filename = 'throughput_comparison_fanin44.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure successfully saved as '{output_filename}'.")

print("\n--- Per-period average throughput ratio (Job1 / Job2) ---")
if len(coarse_throughput1) > 0 and len(coarse_throughput2) > 0:
    # Ensure equal length for per-period comparison
    num_periods = min(len(coarse_throughput1), len(coarse_throughput2))
    for i in range(num_periods):
        # Avoid division-by-zero
        if coarse_throughput2[i] != 0:
            ratio = coarse_throughput1[i] / coarse_throughput2[i]
        else:
            ratio = float('inf')

        print(f"Period {i + 1}:")
        print(f"  - Job1 avg throughput: {coarse_throughput1[i]:.4f} Gbps")
        print(f"  - Job2 avg throughput: {coarse_throughput2[i]:.4f} Gbps")
        print(f"  - Ratio (Job1/Job2): {ratio:.4f}")
