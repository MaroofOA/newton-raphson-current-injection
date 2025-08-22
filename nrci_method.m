% Define the function for the Newton Raphson Current Injection Method for distribution network analysis.
function [v, iteration] = nrci_method(load_data, line_data, slack_bus_voltage, tolerance, max_iter, runs, v_base, s_base)
    % Arguments include the constant slack bus voltage, convergence value, and max. iterations,
    % all of which can be changed when the function is called.

    prompt = {'Ensure the argument data i.e, load and line data are in this format:'
                   'load data: column1 - bus index, column2 - real power (P), and column3 - reactive power (Q)'
                   'line data: column1 - sending bus index, column2 - receiving bus index, column3 - resistance (R), and reactance (X)'
                   'Line data can also have a 5th column if we are dealing with two different voltage levels. In such case, the 5th colum will be the voltage level/base'
                   '\n'
                   'The default value are slack_bus_voltage = 1.5, tolerance = 1e-6, and max_iter = 100'
                   'If you decide to change any of the constant argument value, such as slack_bus_voltage, tolerance, or max_iter, just pass the new value in their respective position'
                   'If you will jump any of the order, kindly put the constant value of the one you are jumping in its position and input the new value of the argument you are changing'
                   'For example: [v, iteration] = bfs_method(load_data, line_data, 1.5, 1e-4), this is because, I want to keep the slack bus the same but want to change the tolerance'
                   'I did not put the value of max_iter because, it is the last one and I want to keep it the same.'
                   };
    disp(prompt);
    user_input = input('Press Enter to continue or type ''quit'' to exit; ', 's');

     % Check if the user wants to quit
    if strcmp(user_input, 'quit')% strcmp is the MATLAB function for comparing two string  first to last
        disp('Exiting the function as per user request probably because the data does not conform with the requirement');
        iteration = []; % Return an empty array
        v = [];
        return;
    end 
    
    if nargin < 3, slack_bus_voltage = 1; end
    if nargin < 4, tolerance = 1e-6; end
    if nargin < 5, max_iter = 100; end
    if nargin < 6, runs = 10; end
    if nargin < 7, v_base = 12.66; end
    if nargin < 8, s_base = 100; end
    
    % System base values
    % Managing the bases of different voltage levels between MV and LV
    if size(line_data, 2) >= 5
        V_base = line_data(:,5);  % per-line voltage base in kV
    else
        V_base = v_base * ones(size(line_data, 1), 1);  % default voltage base
    end
    
    S_base = s_base;  % GLOBAL S_base (in MVA) for the entire system
    Z_base = (V_base.^2) ./ S_base;  % per-line Z_base, consistent S_base

    num_buses = size(load_data, 1); % Number of buses
    P_load = load_data(:, 2) / 100000; % Real power in p.u.
    Q_load = load_data(:, 3) / 100000; % Reactive power in p.u.
      
    % Preallocate arrays to improve performance
    max_voltage_errors = zeros(1, max_iter);
    % cumulative_iter_times = zeros(1, max_iter);
    computation_times = zeros(1, runs);
    Ybus_time = zeros(1, runs);
    current_Jacobian_time_analysis = zeros(1, runs);
    current_Jacobian_time = zeros(runs, max_iter);
    iter_time = zeros(runs, max_iter);
   
    % Begin NR method for multiple runs
    for run = 1:runs
        % Initialize bus voltages
        v = ones(num_buses, 1) + 1j * zeros(num_buses, 1); % Initial guess
        v(1) = slack_bus_voltage; % Slack bus voltage
        
        % Measure the computational time to compute Ybus
        Ybus_start_time = tic;

        % Construct the Y_bus matrix using the helper function script
        Y_bus = build_Y_bus(line_data, num_buses, Z_base);
        
        Ybus_time(run) = toc(Ybus_start_time);

        % Measure the computational time to compute Jacobian
        current_Jacobian_start_time_analysis = tic;

        % Build the Jacobian matrix using the help function script
        current_Jacobian_analysis = build_current_Jacobian(num_buses, - P_load, - Q_load, v, Y_bus); %#ok<NASGU>
        
        current_Jacobian_time_analysis(run) = toc(current_Jacobian_start_time_analysis);

        % Measure the computational time
        start_time = tic;
        
        % Start NR Iterations
        for iteration = 1:max_iter % Iterate up to max_iter times to perform the BFS algorithm.
            % Start timing for the iteration
            iter_start_time = tic;
            
            % Calculate current mismatch vectors
            I_mismatch = calculate_current_mismatch(num_buses, - P_load, - Q_load, v, Y_bus);
            Ii_mismatch_imag = I_mismatch(2:num_buses);
            Ii_mismatch_real = I_mismatch(num_buses+2:end);
            
            % Measure the computational time to compute the current Jacobian
            Jacobian_start_time = tic;

            % Build the Jacobian matrix using the help function script
            current_Jacobian = build_current_Jacobian(num_buses, - P_load, - Q_load, v, Y_bus);
            
            current_Jacobian_time(run, iteration) = toc(Jacobian_start_time);

            % Solve for corrections
            mismatch = [Ii_mismatch_imag; Ii_mismatch_real];
            corrections = current_Jacobian \ mismatch;

            % Update the voltages
            V_mismatch_real = corrections(1:num_buses-1);
            V_mismatch_imag = corrections(num_buses:end);

            V_new_real = real(v(2:end)) + V_mismatch_real;
            V_new_imag = imag(v(2:end)) + V_mismatch_imag;

            v(2:end) = V_new_real + 1j * V_new_imag; % To rectangular form

            % End the iteration time
            iter_time(run, iteration) = toc(iter_start_time);
            
            % Check for max_difference
            max_diff = max(abs(V_mismatch_real+ 1j * V_mismatch_imag));
            
            % Store max voltage difference and cumulative iteration time
            max_voltage_errors(iteration) = max_diff;  % Assign value directly
        
            % if iteration == 1
            %    cumulative_iter_times(iteration) = iter_time;
            % else
            %    cumulative_iter_times(iteration) = cumulative_iter_times(iteration - 1) + iter_time;
            % end

            % Print max voltage difference at an iteration and its respective
            % computation time till that iteration.
            % fprintf('Iteration %d: max voltage difference = %.10f\n', iteration, max_diff);
            % fprintf('Time taken till %d iteration: %.4f seconds\n', iteration, iter_time);
            
            if max_diff <= tolerance % Check if the maximum voltage difference between iterations is less than the tolerance.
                break; % Break the loop if convergence is achieved.
            end
        end
        
        % End computational time measurement
        computation_times(run) = toc(start_time);

    end
    
    % To handle only the iterations that were executed
    actual_iterations = iteration;  % Number of iterations performed until convergence
    max_voltage_errors = max_voltage_errors(1:actual_iterations);  % Adjust size of the errors array
    % cumulative_iter_times = cumulative_iter_times(1:actual_iterations);  % Adjust size of the cumulative times array

    % Prepare results and print them
    fprintf('\nBus Voltages:\n');
    for i = 1:num_buses % Iterate over each bus voltage.
        magnitude = abs(v(i)); % Calculate the magnitude of the voltage.
        phase_angle = rad2deg(angle(v(i))); % Calculate the angle of the voltage in degrees.
        rectangular_form = sprintf('%.4f + %.4fj', real(v(i)), imag(v(i)));
        polar_form = sprintf('%.4f ∠ %.2f°', magnitude, phase_angle);
        fprintf('Bus %d: Rectangular form: %s p.u., Polar form: %s\n', i, rectangular_form, polar_form); % Print the voltage in both forms.
    end
    
    % Compute system loss, substation power, and other results
    % Calculate system loss
    fprintf('\nSystem Losses:\n')
    [total_active_loss_pu, total_reactive_loss_pu] = calculate_system_loss(num_buses, line_data, v, Z_base);
    
    fprintf('\nConverged in %d iterations.\n', iteration); % Print the number of iterations it took to converge.
    
    total_active_loss = total_active_loss_pu * S_base;
    total_reactive_loss = total_reactive_loss_pu * S_base;
    fprintf('\nTotal active power loss = %.4f MW\n',total_active_loss);
    fprintf('Total reactive power loss = %.4f MVar\n', total_reactive_loss);
    
    % Calculate substation power
    substation_active_power = S_base * (sum(P_load) + total_active_loss_pu);
    substation_reactive_power = S_base * (sum(Q_load) + total_reactive_loss_pu);
    fprintf('Substation active power = %.4f MW\n', substation_active_power);
    fprintf('Substation reactive power = %.4f MVar\n', substation_reactive_power);

    % Find minimum and maximum voltages and their corresponding bus indices
    [min_voltage, min_index] = min(abs(v));
    [max_voltage, max_index] = max(abs(v));

    % Print the minimum and maximum voltages along with the bus indices
    fprintf('\nMinimum voltage = %.4f pu at bus %d\n', min_voltage, min_index);
    fprintf('Maximum voltage = %.4f pu at bus %d\n', max_voltage, max_index);

    % Calculate and print the computation time for Ybus
    fprintf('\nThe Ybus was computed in %f secs\n', mean(Ybus_time));

    % Calculate and print the computation time for the current Jacobian
    fprintf('\nThe current Jacobian was computed in %f secs\n', mean(current_Jacobian_time_analysis));

    % compute average time for Jacobian in each iteration per run
    avg_current_Jacobian_time = mean(current_Jacobian_time, 1);
    average_current_Jacobian_time = avg_current_Jacobian_time(1:iteration);
    
    % Calculate and print the computation time for Jacobian
    fprintf('\nThe Jacobian was computed in secs\n')
    disp(average_current_Jacobian_time);

    fprintf('\nAverage Computation time for each iterations after %d runs:\n', run)
    % compute average time for each iteration
    average_iter_time = mean(iter_time, 1);
    for i = 1:iteration
        fprintf('The average iteration time for iteration %d is %f seconds\n', ...
            i, average_iter_time(i));
    end
    
    % Get the cumulative sum of each iteration for plotting
    cumulative_iter_times = cumsum(average_iter_time(1:iteration));

    fprintf('\nCumulative sum of the average of each iteration per run:\n')
    disp(cumulative_iter_times)

    format longG
    fprintf('\nMaximum Voltage Error for each iteration:\n')
    disp(max_voltage_errors)

    % Store the formatted computation times as strings so as to return it
    % as a formated list of string in 4dp.
    formatted_times = arrayfun(@(x) sprintf('%.4f', x), computation_times, 'UniformOutput', false);
    
    % Print the formatted list of computation times
    fprintf('\nComputation times for each run: [%s]\n', strjoin(formatted_times, ', '));   
    
    % Calculate and print the average computation time
    average_time = mean(computation_times);
    fprintf('Average computation time for the model over %d runs: %.4f seconds\n', runs, average_time);

    % Extract the magnitudes of the voltages
    voltage_magnitudes = abs(v);
    voltage_angles = angle(v);
      
    % Create the graph using the provided line data to visualize the system
    % network
    G = digraph(line_data(:, 1), line_data(:, 2));
    % Plot the graph
    fig = figure;
    h = plot(G);
    % Customize the appearance
    h.NodeColor = 'r';  % Node color
    h.EdgeColor = 'b';  % Edge color
    h.ArrowSize = 10;   % Arrow size
    filename1 = sprintf('%d_bus_network.fig', num_buses);
    filename2 = sprintf('%d_bus_network.png', num_buses);
    savefig(fig, fullfile(pwd, 'Figures', filename1));
    exportgraphics(gcf, fullfile(pwd, 'Figures', filename2), 'Resolution', 300);
    close(fig);

    % Create a plot of bus voltages
    fig = figure;
    subplot(1, 2, 1); % 1 row, 2 columns, first subplot
    plot(1:length(voltage_magnitudes), voltage_magnitudes, '-', 'LineWidth', 2);
    set(gca, 'FontName', 'Palatino Linotype'); 
    xlabel('Bus Number', 'FontName', 'Palatino Linotype');
    ylabel('Voltage Magnitude (p.u.)', 'FontName', 'Palatino Linotype');
    title('Bus Voltage Profile', 'FontName', 'Palatino Linotype');
    % grid on;
    
    % Create a plot of bus voltages angle
    subplot(1, 2, 2); % 1 row, 2 columns, second subplot
    plot(1:length(voltage_angles), voltage_angles, '-', 'LineWidth', 2);
    set(gca, 'FontName', 'Palatino Linotype'); 
    xlabel('Bus Number', 'FontName', 'Palatino Linotype');
    ylabel('Voltage Angle (radian)', 'FontName', 'Palatino Linotype');
    title('Bus Voltage Angle', 'FontName', 'Palatino Linotype');
    % grid on;
    filename1 = sprintf('nrci_vol_profile_%d_bus.fig', num_buses);
    filename2 = sprintf('nrci_vol_profile_%d_bus.png', num_buses);
    savefig(fig, fullfile(pwd, 'Figures', filename1));
    exportgraphics(gcf, fullfile(pwd, 'Figures', filename2), 'Resolution', 300);
    close(fig);
    
    % Plot the maximum voltage error vs. computation time till iteration
    fig = figure;
    plot(cumulative_iter_times, max_voltage_errors, '-o', 'LineWidth', 2);
    set(gca, 'FontName', 'Palatino Linotype'); 
    xlabel('Computation Time (seconds)', 'FontName', 'Palatino Linotype');
    ylabel('Maximum Error', 'FontName', 'Palatino Linotype');
    title('Maximum Error vs. Computation Time', 'FontName', 'Palatino Linotype');
    % grid on;
    filename1 = sprintf('nrci_maxE_Comp_%d_bus.fig', num_buses);
    filename2 = sprintf('nrci_maxE_Comp_%d_bus.png', num_buses);
    savefig(fig, fullfile(pwd, 'Figures', filename1));
    exportgraphics(gcf, fullfile(pwd, 'Figures', filename2), 'Resolution', 300);
    close(fig);
    
    % Write results to a text file with UTF-8 encoding
    fid_txt = fopen(sprintf('Result_files/nrci_bus_voltage_%d_bus.txt', num_buses), 'w', 'n', 'UTF-8');
    fprintf(fid_txt, 'Converged in %d iterations.\n\n', iteration);
    fprintf(fid_txt, 'Bus Voltages:\n');
    % Print aligned column headers
    fprintf(fid_txt, '%-12s %-12s %-12s %-12s %-12s\n', ...
        'Bus Number', 'Real', 'Imaginary', 'Magnitude', 'Angle');
    % Print each row of bus voltage data, aligned under headers
    for i = 1:num_buses
        fprintf(fid_txt, '%-12d %-12.4f %-12.4f %-12.4f %-12.2f\n', ...
            i, real(v(i)), imag(v(i)), abs(v(i)), rad2deg(angle(v(i))));
    end
    fclose(fid_txt);
    
    % Write results to a CSV file with writetable, you can also use fopen
    % but everything would be changed from txt to csv
    T = table((1:num_buses)', real(v), imag(v), abs(v), rad2deg(angle(v)), ...
    'VariableNames', {'Bus_Number', 'Real', 'Imaginary', 'Magnitude', 'Angle'});

    writetable(T, sprintf('Result_files/nrci_bus_voltage_%d_bus.csv', num_buses));
    
end