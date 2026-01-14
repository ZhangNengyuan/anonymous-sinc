% MATLAB script to analyze the convergence time of a fluid network model.
% This script runs simulations for different values of the parameter N
% and plots the time required for the window sizes (W1, W2) to converge
% such that |W1 - W2| <= 1.

clear;
clc;
close all;

% --- 1. Define Simulation Parameters ---
% ---------------------------------------
% Define a continuous range of N values to test.
% We use a step of 500 to create a dense set of points for a smoother plot.
N_values = 100:50:2000;

% Initialize a vector to store the convergence time for each N.
convergence_times = zeros(1, length(N_values));

% --- Fixed Model Parameters (from the original script) ---
N_FLOWS = 2;        % Number of flows (W1 and W2)
F = [2, 2];         % Parameter Fi for each flow.
D = [0.823, 0.823]; % Propagation delay for each flow (RTT).
C = 452.8;          % Link capacity (pkts/RTT)
K = 80;             % Queue threshold (pkts)

% --- Simulation Setup ---
T_START = 0;
% Set a long simulation time to ensure convergence can be reached for all N.
T_END = 20000;
t_span = [T_START, T_END];

% --- Initial State ---
% Asymmetric initial conditions to observe convergence.
W_initial = [100, 0]; % Initial window sizes
q_initial = 0.0;      % Initial queue size
history = [W_initial, q_initial]'; % history is the state at t<=0

% --- DDE Solver Options ---
options = ddeset('RelTol', 1e-5);
lags = D;

% --- 2. Run Simulation for Each N Value ---
% ------------------------------------------
fprintf('Starting simulations for different N values...\n');

for i = 1:length(N_values)
    N_PARAM = N_values(i);
    fprintf('Running simulation for N = %d...\n', N_PARAM);

    % Solve the system of DDEs by calling the dde23 solver.
    % The solver uses the 'dde_system' function defined at the end of this script.
    sol = dde23(@(t,y,Z) dde_system(t, y, Z, N_FLOWS, N_PARAM, F, D, C, K), lags, history, t_span, options);

    % Extract solution vectors from the solver output
    t_sol = sol.x;
    y_sol = sol.y;
    
    % Enforce non-negativity on results, as window sizes cannot be negative.
    y_sol = max(0, y_sol); 
    W1_sol = y_sol(1, :);
    W2_sol = y_sol(2, :);

    % Calculate the absolute difference between W1 and W2 over time
    W_diff = abs(W1_sol - W2_sol);

    % Find the first time point where the difference is less than or equal to 1.
    % 'find' returns the index of the first element that meets the condition.
    convergence_idx = find(W_diff <= 1, 1, 'first');

    if ~isempty(convergence_idx)
        % If a convergence point is found, store the corresponding time.
        time_to_converge = t_sol(convergence_idx);
        convergence_times(i) = time_to_converge;
        fprintf('  Convergence found at time t = %.2f RTTs\n', time_to_converge);
    else
        % If no convergence point is found within T_END, store NaN (Not a Number).
        % This indicates the simulation ended before the condition was met.
        convergence_times(i) = NaN;
        fprintf('  Warning: Convergence condition |W1-W2|<=1 not met for N = %d within the simulation time (T_END = %d).\n', N_PARAM, T_END);
    end
end

fprintf('All simulations finished.\n\n');

% --- 3. Visualize the Results ---
% --------------------------------
% Filter out any NaN values to prevent plotting errors.
valid_indices = ~isnan(convergence_times);
plot_N_values = N_values(valid_indices);
plot_times = convergence_times(valid_indices);

if isempty(plot_N_values)
    fprintf('Could not generate plot because no simulations converged.\n');
else
    % Create a new figure for the plot.
    figure('Name', 'Convergence Time Analysis', 'Position', [100, 100, 800, 600]);
    
    % Plot N values vs. the time it took for convergence.
    plot(plot_N_values, plot_times, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    
    % Add titles and labels to the plot as requested.
    title('', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Switch Memory Size','FontSize', 12);
    ylabel('Time for Convergence (RTTs)', 'FontSize', 12);
    grid on;
    box on;
    set(gca, 'FontSize', 10);
    
    fprintf('Plot generated successfully.\n');
end


% --- DDE System Definition (from the original script) ---
% --------------------------------------------------------
function dydt = dde_system(t, y, Z, N_FLOWS, N_PARAM, F, D, C, K)
    % This function defines the system of delay differential equations.
    % It is called by the dde23 solver at each time step.
    %
    % Inputs:
    %   t: current time
    %   y: current state vector [W1(t), W2(t), ..., q(t)]'
    %   Z: matrix of delayed states. Z(:,k) is the state at time t-lags(k)
    %   (Other inputs are the model parameters)
    %
    % Output:
    %   dydt: column vector of derivatives [dW1/dt, dW2/dt, ..., dq/dt]'

    % Unpack current state from the vector y
    W_current = y(1:N_FLOWS);
    q_current = y(N_FLOWS + 1);
    
    % --- Calculate Derivatives ---
    dydt = zeros(N_FLOWS + 1, 1); % Initialize derivative vector

    % a) Calculate dW_i/dt for each flow
    R_current = D' + q_current / C;
    R_current = max(R_current, 1e-6); % Avoid division by zero
    
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
    
    % Assemble the final vector of derivatives
    dydt(1:N_FLOWS) = dW_dt;
    dydt(N_FLOWS + 1) = dq_dt;
end
