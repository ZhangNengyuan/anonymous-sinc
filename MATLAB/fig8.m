% main_stability_analysis_2D_heatmap.m
%
% Purpose:
% This is the main script to analyze and visualize the stability of the limit
% cycle over a range of system parameters (w_bar, F_bar) as a 2D heatmap.
%
% Workflow:
% 1. Setup: Sets fixed parameters, defines the grid for variables w_bar and
%    F_bar, and loads the pre-compiled function handles.
% 2. Resumption: Automatically saves progress, allowing recovery if interrupted.
% 3. Parallel Computing: Uses MATLAB's Parallel Computing Toolbox to speed up
%    the computation.
% 4. Calculation Loop: Iterates over each (w_bar, F_bar) pair and
%    calculates the spectral radius rho.
% 5. Post-processing: Uses interpolation to fill any missing data points (NaNs).
% 6. Visualization: Generates a high-quality 2D heatmap of
%    rho vs. (w_bar, F_bar), where rho is represented by color.

clear; clc; close all;

% --- User Settings ---
use_parallel = true; % Set to true to use parallel computing (recommended)

% --- 1. Setup Parameters ---
fprintf('1. Setting up parameters and loading functions...\n');
try
    load('corrected_system_functions.mat');
catch
    error('Function file not found. Please run "create_symbolic_functions_corrected.m" first.');
end

% Define fixed system parameters
params.n = 8;
params.N = 2000;

% Define variable parameter ranges for high-resolution plotting
F_bar_range = linspace(1, 64, 64);
w_bar_range = linspace(10, 1000, 100);

% --- 2. Resumption and Recovery ---
results_filename = 'intermediate_results_corrected.mat';
if exist(results_filename, 'file')
    fprintf('Intermediate results file found, loading progress...\n');
    load(results_filename, 'rho_matrix', 'loaded_F_range', 'loaded_w_bar_range');
    if ~isequal(loaded_F_range, F_bar_range) || ~isequal(loaded_w_bar_range, w_bar_range)
        fprintf('Warning: Saved parameters do not match current settings. Starting new calculation.\n');
        rho_matrix = NaN(length(F_bar_range), length(w_bar_range));
    end
else
    fprintf('No intermediate results found. Starting new calculation.\n');
    rho_matrix = NaN(length(F_bar_range), length(w_bar_range));
end

% --- 3. Main Calculation Loop ---
fprintf('2. Starting calculation of spectral radius rho...\n');
tic;

has_parallel_toolbox = license('test', 'Parallel_Computing_Toolbox');
if use_parallel && has_parallel_toolbox
    pool = gcp('nocreate');
    if isempty(pool), parpool(); end
    fprintf('Parallel pool ready. Using parfor for calculation.\n');
    
    parfor i = 1:length(F_bar_range)
        temp_rho_row = rho_matrix(i, :);
        current_params = params;
        current_params.F_bar = F_bar_range(i);
        
        for j = 1:length(w_bar_range)
            if ~isnan(temp_rho_row(j)), continue; end
            
            fprintf('Calculating: F_bar = %.3f, w_bar = %.2f\n', F_bar_range(i), w_bar_range(j));
            current_params.w_bar = w_bar_range(j);
            try
                temp_rho_row(j) = calculate_rho_hp_corrected(current_params, F0_handle, F1_handle, J0_handle, J1_handle);
            catch ME
                fprintf('Error at F_bar=%.3f, w_bar=%.2f: %s\n', F_bar_range(i), w_bar_range(j), ME.message);
            end
        end
        rho_matrix(i, :) = temp_rho_row;
        parsave(results_filename, rho_matrix, F_bar_range, w_bar_range);
        fprintf('--- Row %d/%d (F_bar = %.3f) complete. Progress saved. ---\n', i, length(F_bar_range), F_bar_range(i));
    end
    
else
    fprintf('Running in serial mode. This may take a long time.\n');
    for i = 1:length(F_bar_range)
        for j = 1:length(w_bar_range)
            if ~isnan(rho_matrix(i, j)), continue; end
            
            fprintf('Calculating: F_bar = %.3f, w_bar = %.2f\n', F_bar_range(i), w_bar_range(j));
            current_params = params;
            current_params.F_bar = F_bar_range(i);
            current_params.w_bar = w_bar_range(j);
            try
                rho_matrix(i, j) = calculate_rho_hp_corrected(current_params, F0_handle, F1_handle, J0_handle, J1_handle);
            catch ME
                fprintf('Error at F_bar=%.3f, w_bar=%.2f: %s\n', F_bar_range(i), w_bar_range(j), ME.message);
            end
        end
        loaded_F_range = F_bar_range;
        loaded_w_bar_range = w_bar_range;
        save(results_filename, 'rho_matrix', 'loaded_F_range', 'loaded_w_bar_range');
        fprintf('--- Row %d/%d (F_bar = %.3f) complete. Progress saved. ---\n', i, length(F_bar_range), F_bar_range(i));
    end
end

fprintf('All calculations finished. Total time: %.2f minutes.\n', toc/60);

% --- 4. Post-processing: Fill Missing Values ---
fprintf('3. Post-processing data to fill gaps...\n');
[X, Y] = meshgrid(w_bar_range, F_bar_range);
valid_data_mask = ~isnan(rho_matrix);
x_coords = X(valid_data_mask);
y_coords = Y(valid_data_mask);
rho_values = rho_matrix(valid_data_mask);

if sum(valid_data_mask(:)) < 3
    fprintf('Warning: Not enough valid data points for interpolation. Plotting raw data.\n');
    rho_filled = rho_matrix;
else
    % Use griddata for interpolation
    rho_filled = griddata(x_coords, y_coords, rho_values, X, Y, 'cubic');
    % Fill any remaining NaNs that might be outside the convex hull
    remaining_nans = isnan(rho_filled);
    if any(remaining_nans(:))
        rho_filled(remaining_nans) = griddata(x_coords, y_coords, rho_values, X(remaining_nans), Y(remaining_nans), 'nearest');
    end
end

% --- 5. Visualization (MODIFIED FOR 2D HEATMAP) ---
fprintf('4. Generating final 2D heatmap...\n');
figure('Name', 'Limit Cycle Stability (2D Heatmap)', 'Color', 'w');

% Use imagesc to create the 2D plot
% w_bar_range is for the x-axis, F_bar_range is for the y-axis
% rho_filled provides the color data
imagesc(w_bar_range, F_bar_range, rho_filled);

% Set the y-axis direction to be normal (ascending from bottom to top)
set(gca, 'YDir', 'normal');

% Apply a colormap suitable for black and white printing.
% 'flipud(gray)' makes high values dark and low values light.
% Other good options are 'gray' (high values are light), 'parula', or 'viridis'.
colormap(parula);
cbar = colorbar;
ylabel(cbar, '\rho', 'FontWeight', 'bold', 'FontSize', 12); % Add label to colorbar

% Add labels and title
xlabel('W (packets)', 'FontWeight', 'bold');
ylabel('F', 'FontWeight', 'bold');
title('Limit Cycle Stability: \rho vs. (W, F)', 'FontWeight', 'bold');

% General plot aesthetics
grid on;
box on;
set(gca, 'FontSize', 12, 'LineWidth', 1);

% --- 6. Save Final Results ---
final_filename = 'final_stability_results_corrected.mat';
fprintf('Saving final results to %s\n', final_filename);
save(final_filename, 'rho_matrix', 'rho_filled', 'F_bar_range', 'w_bar_range');
if exist(results_filename, 'file'), delete(results_filename); end

fprintf('Process complete.\n');

% --- Helper function for saving within parfor loops ---
function parsave(fname, rho_matrix, F_bar_range, w_bar_range)
    loaded_F_range = F_bar_range;
    loaded_w_bar_range = w_bar_range;
    save(fname, 'rho_matrix', 'loaded_F_range', 'loaded_w_bar_range');
end