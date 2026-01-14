import numpy as np
import matplotlib.pyplot as plt

# --- Part 1: ns3 Data Processing Functions ---

# 时间偏移量，单位为us，初始模拟时间为1s
time_offset = 1000000

def read_throughput_data(filename):
    """
    读取ns-3吞吐量数据文件。
    
    Args:
        filename (str): 数据文件路径。

    Returns:
        一个元组，包含两个numpy数组：时间戳 (μs) 和 吞吐量 (Gbps)。
    """
    time = []
    throughput = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    timestamp_us = int(parts[0])
                    bytes_per_interval = int(parts[2])
                    
                    time.append(timestamp_us - time_offset)
                    
                    # 计算吞吐量: (bytes * 8 bits/byte) / (100us interval) / 1e9 bits/Gbps
                    throughput_gbps = bytes_per_interval * 8 / (100e-6) / 1e9
                    throughput.append(throughput_gbps)
    except FileNotFoundError:
        print(f"警告: 数据文件 '{filename}' 未找到。这部分数据将不会被绘制。")
        return np.array([]), np.array([])
    return np.array(time), np.array(throughput)

def read_cwnd_data(filename):
    """
    读取ns-3拥塞窗口(cwnd)数据文件。
    """
    time = []
    cwnd = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                t, c_size = map(float, line.strip().split())
                time.append(t - time_offset)
                cwnd.append(c_size)
    except FileNotFoundError:
        print(f"警告: 数据文件 '{filename}' 未找到。这部分数据将不会被绘制。")
        return np.array([]), np.array([])
    return np.array(time), np.array(cwnd)

def read_queue_data(filename):
    """
    读取ns-3队列大小数据文件。
    """
    time = []
    queue_size = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                t, q_size = map(float, line.strip().split())
                time.append(t - time_offset)
                queue_size.append(q_size)
    except FileNotFoundError:
        print(f"警告: 数据文件 '{filename}' 未找到。这部分数据将不会被绘制。")
        return np.array([]), np.array([])
    return np.array(time), np.array(queue_size)


# --- Part 2: Fluid Model Simulation (translation of atp.m) ---

# 2.1: 模型参数定义 (基于 luka.py)
N_FLOWS = 2
F = np.array([4.0, 8.0])
D = np.array([0.903, 1.806]) # Lags
C = 412.8
K = 80.0
N_PARAM = 4096 * 16

# 2.2: DDE 系统定义
def dde_system_py(y, history_lookup, t, lags):
    """
    定义DDE方程组。
    """
    W_current = y[:N_FLOWS]
    q_current = y[N_FLOWS]

    p_delayed_for_W = np.zeros(N_FLOWS)
    for k in range(N_FLOWS):
        past_state = history_lookup(t - lags[k])
        q_delayed = past_state[N_FLOWS]
        p_delayed_for_W[k] = 1.0 if q_delayed > K else 0.0

    R_current = D + q_current / C
    R_current = np.maximum(R_current, 1e-9)
    dW_dt = (1.0 / R_current) - (W_current / (2.0 )) * p_delayed_for_W

    W_delayed_for_q = np.zeros(N_FLOWS)
    for j in range(N_FLOWS):
        past_state = history_lookup(t - lags[j])
        W_delayed_for_q[j] = past_state[j]

    total_arrival_rate = 0.0
    for k in range(N_FLOWS):
        prod_term = 1.0
        for j in range(N_FLOWS):
            if j != k:
                prod_term *= (1.0 - W_delayed_for_q[j] / N_PARAM)
        
        prod_term = max(0, prod_term)
        arrival_rate_k = (W_current[k] / R_current[k]) * (prod_term + F[k] * (1 - prod_term))
        total_arrival_rate += arrival_rate_k

    dq_dt = total_arrival_rate - C
    
    dydt = np.zeros(N_FLOWS + 1)
    dydt[:N_FLOWS] = dW_dt
    dydt[N_FLOWS] = dq_dt
    
    return dydt

# 2.3: 简单的DDE求解器
def solve_dde(dde_func, initial_history_vals, t_span, lags, dt=1.0):
    """
    使用前向欧拉法求解DDE。
    减小 'dt' 会提高精度但减慢速度。增大 'dt' 会加快速度但降低精度。
    """
    times = np.arange(t_span[0], t_span[1], dt)
    n_vars = len(initial_history_vals)
    solution = np.zeros((len(times), n_vars))
    
    history_dict = {0: initial_history_vals}
    
    def history_lookup(t):
        if t <= 0:
            return initial_history_vals
        closest_time = max(k for k in history_dict if k <= t)
        return history_dict[closest_time]

    solution[0, :] = initial_history_vals

    for i in range(1, len(times)):
        t_prev = times[i-1]
        y_prev = solution[i-1, :]
        derivs = dde_func(y_prev, history_lookup, t_prev, lags)
        y_next = y_prev + derivs * dt
        y_next = np.maximum(y_next, 0)
        solution[i, :] = y_next
        history_dict[times[i]] = y_next
        
    return times, solution


# --- Part 3: Main Execution and Plotting ---

def main():
    # 3.1 Run fluid model
    print("Running fluid model (DDE solver)...")
    t_span_dde = [0, 1115]
    initial_history = np.array([0.0, 0.0, 0.0], dtype=float)

    simulation_dt = 0.1
    t_dde, sol_dde = solve_dde(dde_system_py, initial_history, t_span_dde, D, dt=simulation_dt)
    print("Fluid model done.")

    W1_sol, W2_sol, q_sol = sol_dde[:, 0], sol_dde[:, 1], sol_dde[:, 2]

    rtt_to_us = 8.858  # from atp.m
    t_plot_dde = t_dde * rtt_to_us

    # 3.2 Read ns-3 data
    print("Reading ns-3 data...")
    base_path = "./"
    path_tp_job1 = base_path + "job1-sinkBytes-p2p-auto.txt"
    path_tp_job2 = base_path + "job2-sinkBytes-p2p-auto.txt"
    path_cwnd_job1 = base_path + "n0-job1-cwnd-p2p-auto.txt"
    path_cwnd_job2 = base_path + "m0-job2-cwnd-p2p-auto.txt"
    path_queue = base_path + "n5-queueSize-p2p-auto.txt"

    time_ns3_tp1, throughput_ns3_1 = read_throughput_data(path_tp_job1)
    time_ns3_tp2, throughput_ns3_2 = read_throughput_data(path_tp_job2)
    time_ns3_cwnd1, cwnd_ns3_1 = read_cwnd_data(path_cwnd_job1)
    time_ns3_cwnd2, cwnd_ns3_2 = read_cwnd_data(path_cwnd_job2)
    time_ns3_q, queue_ns3 = read_queue_data(path_queue)
    print("ns-3 data done.")

    # fluid throughput (convert from pkts/RTT-ish to Gbps using your constant)
    R1_sol = D[0] + q_sol / C
    R2_sol = D[1] + q_sol / C
    throughput1_inst_pr = W1_sol / np.maximum(R1_sol, 1e-9)
    throughput2_inst_pr = W2_sol / np.maximum(R2_sol, 1e-9)

    # NOTE: You used 2144.0 in original code; keep it unchanged.
    # If it's "bits per packet", then pkts_per_RTT * bits_per_pkt / RTT_seconds / 1e9.
    pkts_rtt_to_gbps = 2144.0 / (rtt_to_us * 1e-6 * 1e9)
    throughput1_gbps_dde = throughput1_inst_pr * pkts_rtt_to_gbps
    throughput2_gbps_dde = throughput2_inst_pr * pkts_rtt_to_gbps

    # 3.3 Plot: 1 row, 2 columns
    print("Plotting 1x2 comparison figure...")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))

    colors = {
        "ns3_job1": "#FF7070",
        "ns3_job2": "#7070FF",
        "fluid_job1": "#FF0000",
        "fluid_job2": "#0000FF",
        "ns3_queue": "#70FF70",
        "fluid_queue": "#00AA00",
    }

    # ---- Subplot 1: CWND + Queue (ns-3) and W + q (fluid) ----
    line_ns3_job1 = line_ns3_job2 = line_ns3_q = None

    if time_ns3_cwnd1.size > 0:
        (line_ns3_job1,) = ax1.plot(
            time_ns3_cwnd1, cwnd_ns3_1,
            label="NS3: Job1", linewidth=5, color=colors["ns3_job1"], alpha=0.8
        )
    if time_ns3_cwnd2.size > 0:
        (line_ns3_job2,) = ax1.plot(
            time_ns3_cwnd2, cwnd_ns3_2,
            label="NS3: Job2", linewidth=5, linestyle="--", color=colors["ns3_job2"], alpha=0.8
        )
    if time_ns3_q.size > 0:
        (line_ns3_q,) = ax1.plot(
            time_ns3_q, queue_ns3,
            label="NS3: Queue", linewidth=5, color=colors["ns3_queue"], alpha=0.8
        )

    (line_fluid_job1,) = ax1.plot(
        t_plot_dde, W1_sol, label="Model: Job1", linewidth=5, color=colors["fluid_job1"]
    )
    (line_fluid_job2,) = ax1.plot(
        t_plot_dde, W2_sol, label="Model: Job2", linewidth=5, linestyle="--", color=colors["fluid_job2"]
    )
    (line_fluid_q,) = ax1.plot(
        t_plot_dde, q_sol, label="Model: Queue", linewidth=5, color=colors["fluid_queue"]
    )

    ax1.set_xlabel("Time (μs)", fontsize=35)
    ax1.set_ylabel("CWND (Pkts)", fontsize=35)
    ax1.tick_params(axis="both", labelsize=30)
    ax1.set_xlim(0, 10000)
    ax1.grid(which="major", color="#D3D3D3", linestyle="--", linewidth=0.8)

    # ---- Subplot 2: Throughput (ns-3 vs fluid) ----
    line2_ns3_job1 = line2_ns3_job2 = None
    if time_ns3_tp1.size > 0:
        (line2_ns3_job1,) = ax2.plot(
            time_ns3_tp1, throughput_ns3_1,
            label="NS3: Job1", linewidth=3, color=colors["ns3_job1"], alpha=0.8
        )
    if time_ns3_tp2.size > 0:
        (line2_ns3_job2,) = ax2.plot(
            time_ns3_tp2, throughput_ns3_2,
            label="NS3: Job2", linewidth=3, linestyle="--", color=colors["ns3_job2"], alpha=0.8
        )

    (line2_fluid_job1,) = ax2.plot(
        t_plot_dde, throughput1_gbps_dde, label="Model: Job1", linewidth=3, color=colors["fluid_job1"]
    )
    (line2_fluid_job2,) = ax2.plot(
        t_plot_dde, throughput2_gbps_dde, label="Model: Job2", linewidth=3, linestyle="--", color=colors["fluid_job2"]
    )

    ax2.set_xlabel("Time (μs)", fontsize=35)
    ax2.set_ylabel("Throughput (Gbps)", fontsize=35)
    ax2.tick_params(axis="both", labelsize=30)
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0, 80)
    ax2.grid(which="major", color="#D3D3D3", linestyle="--", linewidth=0.8)


    # Set axes borders black
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    fig.patch.set_visible(False)

    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.18, top=0.75, wspace=0.18)

    output_filename = "fanin44_rtt816_three_comparisons_combined.pdf"
    fig.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="none", edgecolor="none")
    plt.close(fig)

    print(f"Saved: {output_filename}")
    print("Done.")


if __name__ == "__main__":
    main()
