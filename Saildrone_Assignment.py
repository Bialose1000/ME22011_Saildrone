"""
Module for computational modelling of an autonomous sail drone, 
including dynamics and numerical integration.
"""

__all__ = ['get_vals_hydro']

import base64
import numpy as np
import matplotlib.pyplot as plt

# --- DATA IMPORT BLOCK ---
# NOTE: This file (saildrone_hydro.dat) must be in the same directory as this script.
FILENAME = 'saildrone_hydro.dat'

# Import data
_data = open(FILENAME,"r").read()
# Executing this defines the internal function _eval_data
exec(base64.b64decode(_data).decode('utf-8'))


# --- HYDRODYNAMIC INTERFACE ---
def get_vals_hydro(velocity,heading,turn_rate,rudder_angle):
    """
    Get hydrodynamic force and torque values by interpolating measurement data.
    """
    # Calls the internal _eval_data function defined by the 'exec' statement
    force, torque = _eval_data(velocity,heading,turn_rate,rudder_angle)
    return force, torque


# --- NUMERICAL STEP FUNCTIONS ---
def step_euler(state_deriv, t, h, z):
    return z + h * state_deriv(t, z)

def step_rk(state_deriv, t, h, z):
    k1 = state_deriv(t, z)
    k2 = state_deriv(t + h/2, z + h/2 * k1)
    k3 = state_deriv(t + h/2, z + h/2 * k2)
    k4 = state_deriv(t + h,   z + h * k3)
    return z + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# --- NUMERICAL SOLVER ---
def solve_ivp(state_deriv, t0, tmax, dt, z0, method='RK'):
    # Ensure z0 is initialized as a 2D column vector
    z = z0[:, np.newaxis] 
    t = np.array([t0])
    
    n = 0
    while t[n] <= tmax: 
        # Adaptive step size 'h' to align exactly with tmax
        h = min(dt, tmax - t[n])
        if h <= 1e-12:
            break

        z_n = z[:, n] 

        if method == 'Euler':
            znext = step_euler(state_deriv, t[n], h, z_n)
        else:
            znext = step_rk(state_deriv, t[n], h, z_n)

        # Update time and state
        t = np.append(t, t[n] + h)
        z = np.append(z, znext[:, np.newaxis], axis=1)

        n += 1

    return t, z


# --- STATE DERIVATIVE FUNCTION ---
def state_deriv_saildrone(t, z):
    M = 2500.0
    I = 10000.0
    A_sail = 15.0
    L_sail = 0.100
    RHO_AIR = 1.225
    
    Vw = 6.7
    Wind_dir = np.pi

    x, y, theta, vx, vy, omega = z

    # Control Input Schedule
    if t < 60:
        beta_sail_deg = -45.0
        beta_rudder_deg = 0.0
    elif 60 <= t < 65:
        beta_sail_deg = -22.5
        beta_rudder_deg = +2.1
    else:
        beta_sail_deg = -22.5
        beta_rudder_deg = 0.0

    beta_sail = np.deg2rad(beta_sail_deg)
    beta_rudder = np.deg2rad(beta_rudder_deg)

    # Aerodynamic Forces
    V_w = np.array([Vw * np.cos(Wind_dir), Vw * np.sin(Wind_dir)])
    V_drone = np.array([vx, vy])
    
    V_apparent = V_w - V_drone
    V_a_mag = np.linalg.norm(V_apparent)
    
    gamma_a = np.arctan2(V_apparent[1], V_apparent[0]) 
    gamma_sail = theta + beta_sail
    alpha = gamma_sail - gamma_a
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha)) 

    C_D = 1 - np.cos(2 * alpha)
    C_L = 1.5 * np.sin(2 * alpha) + 0.5 * np.sin(2 * theta) 

    q = 0.5 * RHO_AIR * V_a_mag**2
    
    F_D_mag = q * A_sail * C_D
    F_L_mag = q * A_sail * C_L
    
    F_D_x = -F_D_mag * np.cos(gamma_a)
    F_D_y = -F_D_mag * np.sin(gamma_a)
    
    F_L_x = -F_L_mag * np.sin(gamma_a)
    F_L_y = F_L_mag * np.cos(gamma_a)
    
    F_aero_x = F_D_x + F_L_x
    F_aero_y = F_D_y + F_L_y
    
    r_x = -L_sail * np.cos(theta)
    r_y = -L_sail * np.sin(theta)
    
    tau_aero = r_x * F_aero_y - r_y * F_aero_x

    # Hydrodynamic Forces (from external module)
    velocity_vec = np.array([vx, vy])
    heading = theta
    turn_rate = omega
    rudder_angle = beta_rudder
    
    F_hydro_xy, tau_hydro = get_vals_hydro(velocity_vec, heading, turn_rate, rudder_angle)
    
    F_hydro_x = F_hydro_xy[0]
    F_hydro_y = F_hydro_xy[1]
    
    # Total Force and Torque
    F_tot_x = F_aero_x + F_hydro_x
    F_tot_y = F_aero_y + F_hydro_y
    
    tau_tot = tau_aero + tau_hydro
    
    # State Derivatives (dz/dt)
    dz1_x = vx
    dz2_y = vy
    dz3_theta = omega
    
    dz4_vx_dot = F_tot_x / M
    dz5_vy_dot = F_tot_y / M
    dz6_omega_dot = tau_tot / I
    
    return np.array([dz1_x, dz2_y, dz3_theta, dz4_vx_dot, dz5_vy_dot, dz6_omega_dot])


# --- MAIN EXECUTION BLOCK (Tasks 4 & 5) ---
if __name__ == '__main__':
    # Initial conditions (Task 4)
    # z0 = [x, y, theta, vx, vy, omega]
    z0 = np.array([0.0, 0.0, np.pi/2, 0.0, 2.9, 0.0]) # Starts Northbound
    
    # Simulation Parameters
    t0 = 0.0
    tmax = 100.0
    dt = 0.01 
    
    print("Starting Sail Drone simulation (RK4 method)...")
    
    # Solve the ODEs
    t_data, z_data = solve_ivp(
        state_deriv_saildrone, 
        t0,
        tmax,
        dt,
        z0,
        method='RK'
    )
    
    # Extract results
    x = z_data[0, :]
    y = z_data[1, :]
    vx = z_data[3, :]
    vy = z_data[4, :]
    speed = np.sqrt(vx**2 + vy**2)
    
    # --- Plotting (Task 5) ---
    
    # 1. Trajectory Plot
    plt.figure(figsize=(10, 8))
    plt.plot(x, y)
    plt.plot(x[0], y[0], 'go', label='Start')
    plt.plot(x[-1], y[-1], 'rs', label='End')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Sail Drone Trajectory (x-y Plane)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # 2. Speed Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_data, speed)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Sail Drone Speed over Time')
    plt.grid(True)
    plt.show()

    print(f"\nSimulation finished at T={t_data[-1]:.2f} s. Final position: x={x[-1]:.2f} m, y={y[-1]:.2f} m.")