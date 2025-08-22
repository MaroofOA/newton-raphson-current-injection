function Jacobian = build_current_Jacobian(num_buses, P_load, Q_load, v, Y_bus)
    % Function to build the Jacobian matrix for current injection in power flow analysis.
    % Arguments:
    % num_buses - Number of buses in the system
    % P_load - Real power loads at each bus
    % Q_load - Reactive power loads at each bus
    % v - Complex voltage at each bus
    % Y_bus - Admittance matrix (Ybus) for the system
    
    % Initialize the Jacobian sub-matrices J1, J2, J3, and J4 as zero matrices
    J1 = zeros(num_buses - 1);
    J2 = zeros(num_buses - 1);
    J3 = zeros(num_buses - 1);
    J4 = zeros(num_buses - 1);
    
    % Loop over all buses excluding the slack bus (Bus 1)
    for i = 2:num_buses
        % Extract the real and imaginary parts of the voltage at bus i
        real_Vi = real(v(i));
        imag_Vi = imag(v(i));
        
        % Compute the denominator for the diagonal elements
        denom = (real_Vi^2 + imag_Vi^2)^2;

        % Diagonal admittance values (real and imaginary) from the Ybus matrix
        Gii = real(Y_bus(i, i));
        Bii = imag(Y_bus(i, i));
        
        % Compute constants a_i, b_i, c_i, d_i for the diagonal Jacobian terms
        a_i = (Q_load(i) * (real_Vi^2 - imag_Vi^2) - 2 * P_load(i) * real_Vi * imag_Vi) / denom;
        b_i = (P_load(i) * (real_Vi^2 - imag_Vi^2) - 2 * Q_load(i) * real_Vi * imag_Vi) / denom;
        c_i = (P_load(i) * (imag_Vi^2 - real_Vi^2) - 2 * Q_load(i) * real_Vi * imag_Vi) / denom;
        d_i = (Q_load(i) * (real_Vi^2 - imag_Vi^2) - 2 * P_load(i) * real_Vi * imag_Vi) / denom;

        % Compute diagonal elements of the Jacobian matrices
        J1(i - 1, i - 1) = Bii - a_i;  % J1 diagonal element
        J2(i - 1, i - 1) = Gii - b_i;  % J2 diagonal element
        J3(i - 1, i - 1) = Gii - c_i;  % J3 diagonal element
        J4(i - 1, i - 1) = - Bii - d_i;  % J4 diagonal element

        % Off-diagonal elements: Summing over all buses j ≠ i
        for j = 2:num_buses
            if j ~= i
                % Get the off-diagonal elements from Ybus
                Gij = real(Y_bus(i, j));
                Bij = imag(Y_bus(i, j));
                
                % Compute off-diagonal elements of the Jacobian matrices
                J1(i - 1, j - 1) = J1(i - 1, j - 1) + Bij;  % Off-diagonal of J1
                J2(i - 1, j - 1) = J2(i - 1, j - 1) + Gij;  % Off-diagonal of J2
                J3(i - 1, j - 1) = J3(i - 1, j - 1) + Gij;  % Off-diagonal of J3
                J4(i - 1, j - 1) = J4(i - 1, j - 1) - Bij;  % Off-diagonal of J4
            end
        end
    end

    % Form the full Jacobian matrix by combining the submatrices
    Jacobian = [J1, J2; J3, J4];  % Final 2x2 Jacobian matrix
end
