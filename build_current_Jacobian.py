# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:06:25 2024

@author: Morufdeen ATILOLA

"""
import numpy as np

def build_current_Jacobian(num_buses, P_load, Q_load, v, Y_bus):
    """
    Build the Jacobian matrix for current injection in power flow analysis.

    Parameters:
    num_buses : int
        Number of buses in the system.
    P_load : array
        Real power loads at each bus.
    Q_load : array
        Reactive power loads at each bus.
    v : array
        Complex voltage at each bus.
    Y_bus : array
        Admittance matrix (Ybus) of the system.

    Returns:
    Jacobian : array
        Full Jacobian matrix for current injection.
    """

    # Initialize the Jacobian sub-matrices J1, J2, J3, and J4 with zeros
    J1 = np.zeros((num_buses - 1, num_buses - 1), dtype=np.float64)
    J2 = np.zeros((num_buses - 1, num_buses - 1), dtype=np.float64)
    J3 = np.zeros((num_buses - 1, num_buses - 1), dtype=np.float64)
    J4 = np.zeros((num_buses - 1, num_buses - 1), dtype=np.float64)

    # Loop over all buses excluding the slack bus (Bus 1)
    for i in range(1, num_buses):  # Start from bus 2 (index 1)
        # Extract the real and imaginary parts of the voltage at bus i
        real_Vi = np.real(v[i])
        imag_Vi = np.imag(v[i])

        # Compute the denominator for the diagonal elements
        denom = (real_Vi**2 + imag_Vi**2)**2

        # Diagonal admittance values (real and imaginary) from the Ybus matrix
        Gii = np.real(Y_bus[i, i])
        Bii = np.imag(Y_bus[i, i])

        # Compute constants a_i, b_i, c_i, d_i for the diagonal Jacobian terms
        a_i = (Q_load[i] * (real_Vi**2 - imag_Vi**2) - 2 * P_load[i] * real_Vi * imag_Vi) / denom
        b_i = (P_load[i] * (real_Vi**2 - imag_Vi**2) - 2 * Q_load[i] * real_Vi * imag_Vi) / denom
        c_i = (P_load[i] * (imag_Vi**2 - real_Vi**2) - 2 * Q_load[i] * real_Vi * imag_Vi) / denom
        d_i = (Q_load[i] * (real_Vi**2 - imag_Vi**2) - 2 * P_load[i] * real_Vi * imag_Vi) / denom

        # Diagonal elements for the Jacobian matrices
        J1[i - 1, i - 1] = a_i - Bii  # J1 diagonal element
        J2[i - 1, i - 1] = b_i - Gii  # J2 diagonal element
        J3[i - 1, i - 1] = c_i - Gii  # J3 diagonal element
        J4[i - 1, i - 1] = d_i + Bii  # J4 diagonal element

        # Off-diagonal elements: Summing over all buses j != i
        for j in range(1, num_buses):  # Start from bus 2 (index 1)
            if j != i:
                # Get the off-diagonal elements from Ybus
                Gij = np.real(Y_bus[i, j])
                Bij = np.imag(Y_bus[i, j])

                # Off-diagonal elements for the Jacobian matrices
                J1[i - 1, j - 1] -= Bij  # Off-diagonal of J1
                J2[i - 1, j - 1] -= Gij  # Off-diagonal of J2
                J3[i - 1, j - 1] -= Gij  # Off-diagonal of J3
                J4[i - 1, j - 1] += Bij  # Off-diagonal of J4

    # Form the full Jacobian matrix by concatenating the submatrices
    Jacobian = np.block([[J1, J2], [J3, J4]])  # Final Jacobian matrix

    return Jacobian

