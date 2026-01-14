function phase_portrait_dde()
% 这个主函数用于求解并绘制一个延迟微分方程(DDE)系统的相图。
% 该系统模型基于您提供的流体模型方程。
% 此版本会从三个不同的初始条件开始绘制轨迹，并在同一张图上显示。

    % --- 参数设置 ---
    % Parameters
    % 您可以修改这些参数来观察相图的变化。
    w_bar=18;     % 根据图标题设置 (w_bar = 10)
    F_bar = 2;   % 假设 g = F_bar (g = 1/16)
    n = 8;          % 假设值 (n > 1)
    N = 2048;         % 假设值 (必须大于 W_tilde 的最大值)
    tau = 1;        % 延迟时间 (来自 p(t-1))

     % --- 仿真设置 ---
    % Simulation setup
    lags = tau;      % 定义延迟量
    tspan = [0 30000]; % 仿真时间范围，足够长以确保系统达到极限环

    % --- 定义多个初始条件 ---
    % Define multiple initial conditions
    initial_conditions = {
        [0; -0.3],   % 初始值 1 (在极限环内部)
        [0; -0.35],  % 初始值 2 (在极限环外部左侧)
        [0; 0.25]    % 初始值 3 (在极限环外部右侧)
    };
    colors = {'b', 'r', 'g'}; % 为每个轨迹定义颜色
    legend_entries = {}; % 用于存储图例条目

    % --- 绘制相图 ---
    % Plot the phase portrait
    figure('Name', 'Phase Portrait with Multiple Initial Conditions', 'Color', 'white');
    hold on; % 保持图形，以便在同一张图上绘制所有轨迹

    % --- 循环遍历每个初始条件 ---
    for k = 1:length(initial_conditions)
        history = initial_conditions{k};
        color = colors{k};
        
        % --- 求解延迟微分方程 ---
        % Solve the DDE
        ddefun_handle = @(t,y,Z) dde_system(t, y, Z, w_bar, F_bar, n, N);
        sol = dde23(ddefun_handle, lags, history, tspan);

        % 绘制轨迹
        plot(sol.y(1,:), sol.y(2,:), '-', 'Color', color, 'LineWidth', 1.5);
        legend_entries{k} = sprintf('Start: (%.1f, %.1f)', history(1), history(2));
        
        % --- 在轨迹上添加方向箭头 (已注释掉) ---
        % Add direction arrows to the trajectory (Commented out)
        % 在轨迹的后半段均匀选取一些点来绘制箭头
        % t_eval = linspace(sol.x(end) * 0.7, sol.x(end), 15); 
        % y_eval = deval(sol, t_eval);

        % 计算这些点上的导数 (即向量场的方向)
        % dydt_eval = zeros(2, length(t_eval));
        % for i = 1:length(t_eval)
        %     y_lag = deval(sol, t_eval(i) - tau);
        %     dydt_eval(:,i) = dde_system(t_eval(i), y_eval(:,i), y_lag, w_bar, F_bar, n, N);
        % end

        % 使用 quiver 函数绘制箭头
        % quiver(y_eval(1,:), y_eval(2,:), dydt_eval(1,:), dydt_eval(2,:), 0.4, 'Color', color, 'LineWidth', 1, 'MaxHeadSize', 0.5);
    end
    
    hold off; % 结束在同一张图上绘图

    % --- 格式化图形 ---
    % Format the plot
    xlabel('$\tilde{W}(t)$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('$\tilde{q}(t)$', 'Interpreter', 'latex', 'FontSize', 14);
    title('Phase Diagram Showing Limit Cycle for Multiple Initial Conditions', 'FontSize', 16);
    grid on;
    legend(legend_entries, 'Location', 'northeast', 'FontSize', 10);
    % 设置坐标轴范围，以匹配您提供的图像
    axis([6 14 -0.4 0.4]);
    set(gca, 'FontSize', 12);

end

% --- DDE 方程定义 ---
% DDE function definition
function dydt = dde_system(t, y, Z, w_bar, F_bar, n, N)
    % 这个函数定义了微分方程组。
    % 输入:
    %   t: 当前时间
    %   y: 当前状态向量 [W_tilde(t); q_tilde(t)]
    %   Z: 延迟状态向量 [W_tilde(t-tau); q_tilde(t-tau)]
    %   w_bar, F_bar, n, N: 系统参数
    % 输出:
    %   dydt: 导数向量 [dW_tilde/dt; dq_tilde/dt]

    W_tilde = y(1);
    q_tilde = y(2);
    
    % 获取延迟的 q_tilde 值
    q_tilde_delayed = Z(2);
    
    % 根据 q_tilde(t-1) 的值计算 p(t-1)
    p_delayed = (q_tilde_delayed > 0);
    
    % 检查分母 (1 + q_tilde) 是否接近零，以避免计算错误
    if abs(1 + q_tilde) < 1e-6
        dydt = [NaN; NaN]; % 返回无效值，求解器会处理
        return;
    end
    
    % --- 计算 dW/dt ---
    dWdt = (1 / (1 + q_tilde)) * (1 - (W_tilde / 2) * p_delayed);
    
    % --- 计算 dq/dt ---
    term1 = (n * W_tilde) / (w_bar * (1 + q_tilde));
    
    % 检查 (1 - W/N) 是否为负，以避免产生复数
    base = 1 - W_tilde / N;
    if base < 0
        % 如果 W > N, 模型可能失效。为避免计算错误，将此项设为0。
        base_pow_n_minus_1 = 0;
    else
        base_pow_n_minus_1 = base^(n - 1);
    end
    
    term2 = base_pow_n_minus_1;
    term3 = F_bar * (1 - base_pow_n_minus_1);
    
    dqdt = term1 * (term2 + term3) - 1;
    
    % 返回导数向量
    dydt = [dWdt; dqdt];
end
