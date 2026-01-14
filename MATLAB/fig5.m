% MATLAB script for simulating a fluid model of network traffic using the
% MATLAB DDE solver 'dde23'.
% This version calculates and displays the throughput for each job and
% generates a second plot matching the user's reference image.
clear;
clc;
close all;
% --- 1. Model Parameter Definition ---
% ------------------------------------
N_FLOWS = 2;          % Number of flows (W1 and W2)
N_PARAM = 4096*16;       % Parameter N in equation (2) (pkts)
% -- Flow-specific parameters --
F = [8, 8];           % Parameter Fi for each flow.
D = [0.95, 0.95];      % Propagation delay for each flow (RTT).
% -- Shared parameters --
C = 2.1;             % Link capacity (pkts/RTT)100Gbps
K = 2;             % Queue threshold (pkts)
% --- 2. Simulation Setup for DDE Solver ---
% ------------------------------------
T_START = 0;
% Adjusted T_END to make the plot's time axis match the reference image (~10000 us)
T_END = 2500;
t_span = [T_START, T_END]; % Time interval for the solver
% --- 3. Initialize State for DDE Solver ---
% ------------------------------------
% lags must be a vector containing all unique delays in the system.
lags = D;
% Set initial conditions (t=0) as a column vector.
W_initial = [0, 0]; % Initial window sizes
q_initial = 0.0;       % Initial queue size
history = [W_initial, q_initial]'; % history is the state at t<=0
% Set solver options.
options = ddeset('RelTol', 1e-5);
% --- 4. Solve the System of DDEs ---
% ------------------------------------
disp('Starting DDE solver...');
sol = dde23(@(t,y,Z) dde_system(t, y, Z, N_FLOWS, N_PARAM, F, D, C, K), lags, history, t_span, options);
disp('Solver finished.');
% --- 5. Extract and Process Results ---
% ------------------------------------
t_sol = sol.x;
y_sol = sol.y;
y_sol = max(0, y_sol); % Enforce non-negativity
W1_sol = y_sol(1, :);
W2_sol = y_sol(2, :);
q_sol = y_sol(N_FLOWS + 1, :);

% --- 6. Calculate Throughput ---
% --------------------------------
% Calculate instantaneous RTT for each flow
R1_sol = D(1) + q_sol / C;
R2_sol = D(2) + q_sol / C;
% Calculate instantaneous throughput for each flow (pkts/RTT)
throughput1_inst_pr = W1_sol ./ R1_sol; % pkts/RTT
throughput2_inst_pr = W2_sol ./ R2_sol; % pkts/RTT
% Calculate average throughput using trapezoidal integration for accuracy
total_time = t_sol(end) - t_sol(1);
avg_throughput_1 = trapz(t_sol, throughput1_inst_pr) / total_time;
avg_throughput_2 = trapz(t_sol, throughput2_inst_pr) / total_time;
% Display results in the command window
fprintf('\n--- Simulation Results ---\n');
fprintf('Job 1 Average Throughput: %.2f pkts/RTT\n', avg_throughput_1);
fprintf('Job 2 Average Throughput: %.2f pkts/RTT\n', avg_throughput_2);
fprintf('--------------------------\n');

% --- 7. Visualize Results ---
% ------------------------------------
rtt_to_us = 8.858;%us/RTT
t_plot = t_sol * rtt_to_us; % Convert time axis to microseconds
% --- Plot 1: Congestion Window and Queue Size (As before) ---
figure('Name', 'Fluid Model Simulation Results (dde23)', 'Position', [100, 100, 800, 600]);
hold on;
plot(t_plot, W2_sol, 'r-', 'LineWidth', 1.5, 'DisplayName', 'job2(cwnd)');
plot(t_plot, W1_sol, 'b--', 'LineWidth', 1.5, 'DisplayName', 'job1(cwnd)');
plot(t_plot, q_sol, 'Color', [0 0.7 0], 'LineStyle', '-', 'LineWidth', 1.5, 'DisplayName', 'Queue Size');
hold off;
title('Congestion Window and Queue Size', 'FontSize', 16);
xlabel('Time (us)', 'FontSize', 12);
ylabel('Size (packets)', 'FontSize',12);
legend('show', 'Location', 'northwest', 'FontSize', 12);
xlim([0, T_END * rtt_to_us]);
ylim([0, 5]);
grid on;
box on;
% --- Plot 2: Throughput Comparison (Modified to match reference) ---
% Conversion factor from pkts/RTT to Gbps
% Assumption: 1 packet = 1500 bytes = 12000 bits
% 1 RTT unit = 9.72 us = 9.72e-6 s
pkts_rtt_to_gbps = (2145.9) / (rtt_to_us * 1e-6 * 1e9); % (bits/pkt) / (s/RTT * bits/Gb)
throughput1_gbps = throughput1_inst_pr * pkts_rtt_to_gbps;
throughput2_gbps = throughput2_inst_pr * pkts_rtt_to_gbps;

figure('Name', 'Throughput Comparison', 'Position', [150, 150, 800, 600]);
set(gca, 'Color', [0.95 0.95 0.95]); % Set background color to light gray like reference
hold on;
% Plotting according to the reference image style
plot(t_plot, throughput1_gbps, 'r-', 'LineWidth', 1.5, 'DisplayName', 'job1');
plot(t_plot, throughput2_gbps, 'b--', 'LineWidth', 1.5, 'DisplayName', 'job2');
hold off;
title('Throughput Comparison', 'FontSize', 16);
xlabel('Time (us)', 'FontSize', 12);
ylabel('Throughput (Gbps)', 'FontSize', 12);
legend('show', 'Location', 'northwest', 'FontSize', 12);
xlim([0, T_END * rtt_to_us]);
ylim([0, 10]); % Set Y-axis limit to match reference image
grid on;
box on;

% --- DDE System Definition ---
function dydt = dde_system(t, y, Z, N_FLOWS, N_PARAM, F, D, C, K)
    % y: current state vector [W1(t), W2(t), ..., q(t)]'
    % Z: matrix of delayed states. Z(:,k) is the state at time t-lags(k)
    % Unpack current state
    W_current = y(1:N_FLOWS);
    q_current = y(N_FLOWS + 1);
    % --- Calculate Derivatives ---
    dydt = zeros(N_FLOWS + 1, 1); % Initialize derivative vector
    % a) Calculate dW_i/dt for each flow
    R_current = D' + q_current / C;
    R_current = max(R_current, 1e-6);
    p_delayed_for_W = zeros(N_FLOWS, 1);
    for k = 1:N_FLOWS
        q_delayed = Z(N_FLOWS + 1, k); 
        if q_delayed > K
            p_delayed_for_W(k) = 1.0;
        else
            p_delayed_for_W(k) = 0.0;
        end
    end
    dW_dt = (1.0 ./ R_current) - (W_current ./ (2.0 .* R_current)) .* p_delayed_for_W;
    % b) Calculate dq/dt for the shared queue
    W_delayed_for_q = zeros(N_FLOWS, 1);
    for j = 1:N_FLOWS
        W_delayed_for_q(j) = Z(j, j); 
    end
    
    total_arrival_rate = 0.0;
    for k = 1:N_FLOWS
        prod_term = 1.0;
        for j = 1:N_FLOWS
            if j ~= k
                prod_term = prod_term * (1.0 - W_delayed_for_q(j) / N_PARAM);
            end
        end
        prod_term = max(0, prod_term);
        arrival_rate_k = (W_current(k) / R_current(k)) * (prod_term + F(k) * (1 - prod_term));
        total_arrival_rate = total_arrival_rate + arrival_rate_k;
    end
    dq_dt = total_arrival_rate - C;
    dydt(1:N_FLOWS) = dW_dt;
    dydt(N_FLOWS + 1) = dq_dt;
end