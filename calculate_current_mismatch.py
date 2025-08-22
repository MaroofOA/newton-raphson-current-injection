# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:04:12 2024

@author: Morufdeen ATILOLA
"""

import numpy as np

def calculate_current_mismatch(num_buses, P_load, Q_load, v, Y_bus):
    """
    Calculate the real and imaginary parts of the current mismatch, interleaved.

    Parameters:
    P_load - Array of real power loads at each bus
    Q_load - Array of reactive power loads at each bus
    v - Array of bus voltages (complex)
    Y_bus - Admittance matrix (complex)
    
    Returns:
    I_mismatch - Interleaved imaginary and real parts of current mismatch (array)
    """
    
    # Initialize the combined current mismatch array
    I_mismatch = np.zeros(2 * num_buses)
    
    for i in range(num_buses):
        # Extract real and imaginary parts of Vi
        imag_Vi = np.imag(v[i])
        real_Vi = np.real(v[i])
        
        # Precompute denominator (real(Vi)^2 + imag(Vi)^2)
        denom = real_Vi**2 + imag_Vi**2
        
        # Compute scheduled current for the imaginary part
        Ii_mismatch_imag = (P_load[i] * imag_Vi - Q_load[i] * real_Vi) / denom
        
        # Compute scheduled current for the real part
        Ii_mismatch_real = (P_load[i] * real_Vi + Q_load[i] * imag_Vi) / denom
        
        # Calculate the current mismatch using Y_bus and Vj
        for j in range(num_buses):
            Gij = np.real(Y_bus[i, j])
            Bij = np.imag(Y_bus[i, j])
            Vj_real = np.real(v[j])
            Vj_imag = np.imag(v[j])

            Ii_mismatch_imag -= (Gij * Vj_imag + Bij * Vj_real)
            Ii_mismatch_real -= (Gij * Vj_real - Bij * Vj_imag)
        
        # Assign the imaginary and real parts interleaved in the output array
        I_mismatch[i] = Ii_mismatch_imag   # Imaginary part first (index 2*i)
        I_mismatch[num_buses + i] = Ii_mismatch_real  # Real part second (index 2*i + 1)
    
    return I_mismatch
