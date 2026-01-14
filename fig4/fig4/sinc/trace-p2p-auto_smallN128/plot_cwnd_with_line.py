import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 时间偏移量，单位为us，初始模拟时间为1s
time_offset = 1000000

def read_cwnd_data(filename):
    time = []
    cwnd = []
    with open(filename, 'r') as f:
        for line in f:
            t, c_size = map(float, line.strip().split())
            # Convert microseconds to RTT numbers (1 RTT = 10 microseconds)
            time.append(t - time_offset)
            cwnd.append(c_size)
    return np.array(time), np.array(cwnd)

def read_queue_data(filename):
    time = []
    queue_size = []
    with open(filename, 'r') as f:
        for line in f:
            t, q_size = map(float, line.strip().split())
            time.append(t - time_offset)
            queue_size.append(q_size)
    
    return np.array(time), np.array(queue_size)

plt.style.use('seaborn-v0_8-whitegrid')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # Fix for displaying minus sign

# Create figure with single y-axis
fig, ax1 = plt.subplots(1, 1, figsize=(12.5, 7))

# Read data for both flows
time_n0, cwnd_n0 = read_cwnd_data('n0-job1-cwnd-p2p-auto.txt')
time_m0, cwnd_m0 = read_cwnd_data('m0-job2-cwnd-p2p-auto.txt')

# Read queue data
#time_q, queue_size = read_queue_data('atp-result/aaa_result/2_convergence/sinc/trace-p2p-auto_smallN128/n5-queueSize-p2p-auto.txt')

# Plot all data on single y-axis
ax1.plot(time_n0, cwnd_n0, label='Job1 (cwnd)', linewidth=1, color='#FF0000')
ax1.plot(time_m0, cwnd_m0, label='Job2 (cwnd)', linewidth=1, dashes=(10,10), color='#0000FF')
#ax1.plot(time_q, queue_size, label='Queue Size', linewidth=2, color='#00AA00')

# Set plot properties
ax1.set_xlabel('Time (μs)', fontsize=35)
ax1.set_ylabel('CWND (Pkts)', fontsize=35)

ax1.grid(False)
ax1.tick_params(axis='both', labelsize=30)

# Set x-axis range
max_time = max(np.max(time_n0), np.max(time_m0), 10000)

# Add horizontal dashed lines at y=114 and y=57
ax1.axhline(y=114, color='black', linestyle='--')
ax1.axhline(y=57, color='black', linestyle='--')

# Add custom y-ticks to ensure 57 and 114 are visible
current_yticks = ax1.get_yticks()
custom_yticks = np.append(current_yticks, [])
ax1.set_yticks(np.unique(custom_yticks))

# Add vertical dashed lines at x=0 and x=2802
ax1.axvline(x=0, color='black', linestyle='--')
ax1.axvline(x=2802, color='black', linestyle='--')

# Add a double-headed arrow between the vertical lines
y_arrow_pos = max(np.max(cwnd_n0), np.max(cwnd_m0)) * 0.9
ax1.annotate('', xy=(0, y_arrow_pos), xytext=(2802, y_arrow_pos),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax1.annotate('2802μs', xy=(2802/2, y_arrow_pos),
             ha='center', va='center',
             bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'),
             fontsize=30, color='black')

# Adjust x-axis to make the vertical line at x=0 visible and hide negative ticks
ax1.set_xlim(-max_time * 0.03, max_time)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x)}' if x >= 0 else ''))
ax1.set_ylim(bottom=0)
ax1.grid(which='major', color='#D3D3D3', linestyle='--', linewidth=0.8)
x_center = max_time / 2
ax1.text(x_center, 180, '114', ha='center', va='center', fontsize=30, color='black')
ax1.text(x_center, 25, '57', ha='center', va='center', fontsize=30, color='black')

# Add legend

for spine in ax1.spines.values():
        spine.set_edgecolor('black')

# Adjust layout and save with high DPI
plt.tight_layout()
plt.savefig('sinc_cwnd_comparison_smallN_line.pdf', dpi=300, bbox_inches='tight')
plt.close() 