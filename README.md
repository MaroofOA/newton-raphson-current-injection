# Newton Raphson Current Injection Method

This repository contains implementations of the **Newton Raphson Current Injection Method** for performing power flow analysis on distribution or transmission networks, provided in both MATLAB and Python. This method is designed to solve power flow equations by iteratively updating the bus voltage based on current injections, making it particularly suitable for systems with varying voltage levels across buses.

## Overview

The Newton Raphson Current Injection Method calculates the bus voltages of a distribution network by considering both real and reactive current injections at each bus. The method converges iteratively by updating the voltage at each bus until the difference between successive iterations meets the specified tolerance level.

### Scope and Conditions

The method is applicable to radial or meshed distribution systems and requires specific conditions:
- **Convergence Tolerance**: The method will continue iterating until the mismatch in current injections is less than the defined tolerance. This ensures the stability and accuracy of the results.
- **Maximum Iterations**: To prevent excessive computation, a maximum number of iterations can be set, after which the algorithm will stop, indicating if convergence was not achieved.
- **Slack Bus Voltage**: The slack bus acts as a reference with a fixed voltage (typically set to 1 p.u.), while other buses are initialized at 1 p.u. for both real and imaginary parts of the voltage.

## Functionality

The function includes the following features:
- **Data Input Requirements**:
  - The **Load Data** must have three columns: bus index, real power (P), and reactive power (Q).
  - The **Line Data** should have four columns: sending bus index, receiving bus index, resistance (R), and reactance (X).
  
- **Customizable Parameters**:
  - **Slack Bus Voltage**: Fixed reference voltage, typically set at 1 p.u.
  - **Convergence Tolerance**: Determines the stopping criterion for iterative updates based on current mismatch (default is `1E-6`).
  - **Maximum Iterations**: Limits the number of iterations (default is 100).

### Usage Guide

1. **Input Format**:
   - **Load Data**:
     - Column 1: Bus index
     - Column 2: Real power (P) in MW
     - Column 3: Reactive power (Q) in MVAR
   - **Line Data**:
     - Column 1: Sending bus index
     - Column 2: Receiving bus index
     - Column 3: Resistance (R) in Ohms
     - Column 4: Reactance (X) in Ohms

2. **Function Call**:
   - Use the load and line data with optional arguments for slack bus voltage, tolerance, and maximum iterations. Defaults apply if optional parameters are not provided.
   - **Python**:
     ```python
     v, iteration = nrci_method(load_data, line_data, slack_bus_voltage=1.0, tolerance=1e-6, max_iterations=100)
     ```
   - **MATLAB**:
     ```matlab
     [v, iteration] = nrci_method(load_data, line_data, 1.0, 1e-6, 100);
     ```

3. **Running the Function**:
   - To exit the function after execution, the user may press Enter or type ‘quit’ to terminate.

### Expected Outputs

Upon successful execution, the function provides:
- **Voltage Profiles**: Final voltage magnitudes and angles for each bus after each iteration.
- **Number of Iterations**: The iteration count until convergence.
- **Timing Information**: Execution time per iteration and average time for the total computation.
- **Power Loss**: Total active and reactive power losses.
- **Visuals**: Graph of voltage magnitude and angle across buses, single line diagram of the network, and a graph of maximum error vs. computational time.
- **Exported Data**: Output voltage profiles saved as `.csv` and `.txt` files.

## Files

- `nrci_method.py`: Python code for the Newton Raphson Current Injection Method for power flow analysis.
- `nrci_method.m`: MATLAB code for the Newton Raphson Current Injection Method for power flow analysis.
