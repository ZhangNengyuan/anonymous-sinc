import numpy as np
import matplotlib.pyplot as plt
import os

# --- 数据文件 ---
throughput_file1 = 'job1-sinkBytes-p2p-auto.txt'
throughput_file2 = 'job2-sinkBytes-p2p-auto.txt'

# 检查所需文件是否存在
required_files = [throughput_file1, throughput_file2]
for f in required_files:
    if not os.path.exists(f):
        print(f"错误：找不到数据文件 '{f}'。请将所需的数据文件与此脚本放在同一目录中。")
        # 创建一个空文件以避免程序崩溃
        open(f, 'a').close()

# 时间偏移量，单位为us，初始模拟时间为1s
time_offset = 1000000

def read_throughput_data(filename):
    """
    读取文本文件并计算吞吐量
    返回时间戳 (us) 和吞吐量 (Gbps)
    原始数据的时间粒度是100us
    """
    time = []
    throughput = []
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.array([]), np.array([]) # 返回默认值
        
    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    timestamp_us = int(parts[0])
                    bytes_per = int(parts[2])
                    time.append(timestamp_us - time_offset)
                    # 计算吞吐量: bytes * 8 bits/byte / (100us * 10^-6 s/us) / 10^9 bits/Gbps
                    throughput_gbps = bytes_per * 8 / 100e-6 / 1e9
                    throughput.append(throughput_gbps)
            except (ValueError, IndexError):
                print(f"警告：跳过文件 '{filename}' 中的无效行: {line.strip()}")

    if not time:
        return np.array([]), np.array([])
    return np.array(time), np.array(throughput)

def calculate_custom_coarse_throughput(time, throughput):
    """
    根据预定义的非均匀周期计算粗粒度吞吐量。
    周期定义 (基于101个点的索引): [0-35], [36-51], [52-68], [69-84], [85-99]
    该函数会将这些索引比例应用到实际的数据点数量上。
    """
    if len(throughput) == 0:
        return np.array([]), np.array([])
        
    coarse_time = []
    coarse_throughput = []
    
    num_points = len(throughput)
    # 基于101个点的周期定义，我们将其转换为比例
    # 边界点为: 0, 36, 52, 69, 85, 100
    period_boundaries = [0, 34, 51, 68, 78, 100]
    
    # 将比例应用到实际数据点数量上，得到分割点索引
    scaled_boundaries = [int(b / 100.0 * num_points) for b in period_boundaries]

    for i in range(len(scaled_boundaries) - 1):
        start_index = scaled_boundaries[i]
        # 确保最后一个周期包含所有剩余的点
        if i == len(scaled_boundaries) - 2:
            end_index = num_points
        else:
            end_index = scaled_boundaries[i+1]

        if start_index >= end_index:
            continue

        time_slice = time[start_index:end_index]
        throughput_slice = throughput[start_index:end_index]
        
        if len(throughput_slice) > 0:
            # 计算该周期的平均吞吐量
            avg_throughput = np.mean(throughput_slice)
            # 将该点放置在周期的平均时间位置
            avg_time = np.mean(time_slice)
            
            coarse_time.append(avg_time)
            coarse_throughput.append(avg_throughput)
            
    return np.array(coarse_time), np.array(coarse_throughput)

# --- 主绘图逻辑 ---
print("正在生成吞吐量对比图...")
plt.style.use('seaborn-v0_8-whitegrid')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 关键修改：统一 figsize
fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))

# 读取数据
time1, throughput1 = read_throughput_data(throughput_file1)
time2, throughput2 = read_throughput_data(throughput_file2)

# 计算粗粒度数据
coarse_time1, coarse_throughput1 = calculate_custom_coarse_throughput(time1, throughput1)
coarse_time2, coarse_throughput2 = calculate_custom_coarse_throughput(time2, throughput2)

coarse_window_us = 1000 

# --- 绘图 ---
# zorder 值越大，越绘制在上层
if len(time1) > 0:
    ax.plot(time1, throughput1, label='Job1 (100us)', linewidth=1.5, color='#FF7070', alpha=0.6, zorder=1)
if len(time2) > 0:
    ax.plot(time2, throughput2, label='Job2 (100us)', linewidth=1.5, dashes=(10, 5), color='#7070FF', alpha=0.6, zorder=1)

if len(coarse_time1) > 0:
    ax.plot(coarse_time1, coarse_throughput1, label=f'Job1 (Periodic Avg)', linewidth=2.5, color='#E60000', zorder=2,marker='^', markersize=8)
if len(coarse_time2) > 0:
    ax.plot(coarse_time2, coarse_throughput2, label=f'Job2 (Periodic Avg)', linewidth=2.5, dashes=(10, 5), color='#0000B3', zorder=2,marker='^', markersize=8)

# --- 设置图表属性 (与cwnd图对齐) ---
# 关键修改：统一字体大小和间距
ax.set_xlabel('Time (us)', fontsize=35)
ax.set_ylabel('Throughput (Gbps)', fontsize=35)

ax.grid(False)
ax.grid(which='major', color='#D3D3D3', linestyle='--', linewidth=0.8)
# 关键修改：统一刻度参数
ax.tick_params(axis='both', labelsize=30)

# 设置坐标轴范围
max_time = 0
if len(time1) > 0: max_time = max(max_time, np.max(time1))
if len(time2) > 0: max_time = max(max_time, np.max(time2))
ax.set_xlim(0, max(max_time, 10000)) # 确保最小范围
ax.set_ylim(bottom=0)

# 关键修改：统一边框样式
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)


# 保存图表
output_filename = 'throughput_comparison_fanin48.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"图表 '{output_filename}' 已保存。")