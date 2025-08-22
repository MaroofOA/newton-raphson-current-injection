function I_mismatch = calculate_current_mismatch(num_buses, P_load, Q_load, v, Y_bus)
    % Inputs:
    % P_load - Vector of real power loads at each bus
    % Q_load - Vector of reactive power loads at each bus
    % v - Vector of bus voltages (complex)
    % Y_bus - Admittance matrix (complex)
    
    % Initialize the combined current mismatch vector
    I_mismatch = zeros(2 * num_buses, 1);
    
    for i = 1:num_buses
        % Extract real and imaginary parts of Vi
        imag_Vi = imag(v(i));
        real_Vi = real(v(i));
        
        % Precompute denominator (real(Vi)^2 + imag(Vi)^2)
        denom = real_Vi^2 + imag_Vi^2;
        
        % Compute scheduled current for the imaginary part
        Ii_mismatch_imag = (P_load(i) * imag_Vi - Q_load(i) * real_Vi) / denom;
        
        % Compute scheduled current for the real part
        Ii_mismatch_real = (P_load(i) * real_Vi + Q_load(i) * imag_Vi) / denom;
        
        % Calculate the current mismatch using Y_bus and Vj
        for j = 1:num_buses
            Gij = real(Y_bus(i, j));
            Bij = imag(Y_bus(i, j));
            Vj_real = real(v(j));
            Vj_imag = imag(v(j));

            Ii_mismatch_imag = Ii_mismatch_imag - (Gij * Vj_imag + Bij * Vj_real);
            Ii_mismatch_real = Ii_mismatch_real - (Gij * Vj_real - Bij * Vj_imag);
        end
        
        % Assign the imaginary and real parts interleaved in the output vector
        % Imaginary part (for bus i) goes in the first num_buses slots
        I_mismatch(i) = Ii_mismatch_imag;
        
        % Real part (for bus i) goes in the next num_buses slots
        I_mismatch(num_buses + i) = Ii_mismatch_real;
    end
end
