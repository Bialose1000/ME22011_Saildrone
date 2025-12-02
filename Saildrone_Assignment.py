import numpy as np
import base64
import matplotlib.pyplot as plt

FILENAME = 'saildrone_hydro.dat'
_data = open(FILENAME,"r").read()
exec(base64.b64decode(_data).decode('utf-8'))

def get_vals(velocity,heading,turn_rate,rudder_angle):
    force, torque = _eval_data(velocity,heading,turn_rate,rudder_angle)
    return force, torque

M = 2500.0
I = 10000.0
A = 15.0
d_A = 0.1
rho_air = 1.225
v_W = np.array([-6.7, 0.0])

def get_control_inputs(t):
    if 0 <= t < 60:
        return np.deg2rad(-45.0), np.deg2rad(0.0)
    elif 60 <= t < 65:
        return np.deg2rad(-22.5), np.deg2rad(2.1)
    else:
        return np.deg2rad(-22.5), np.deg2rad(0.0)

def sail_drone_dynamics(t, S):
    x, y, theta, x_dot, y_dot, theta_dot = S
    S_dot = np.zeros(6)
    S_dot[0:3] = S[3:6]

    beta_sail, beta_rudder = get_control_inputs(t)
    v_D = np.array([x_dot, y_dot])

    F_H, tau_H = get_vals(v_D, theta, theta_dot, beta_rudder)
    F_H_x, F_H_y = F_H

    v_A = v_W - v_D
    v_a = np.linalg.norm(v_A)

    F_A_x = F_A_y = tau_A = 0.0

    if v_a > 1e-6:
        phi_A = np.arctan2(v_A[1], v_A[0])
        phi_sail = theta + beta_sail
        alpha = np.arctan2(np.sin(phi_sail - phi_A), np.cos(phi_sail - phi_A))

        C_D = 1.0 - np.cos(2.0 * alpha)
        C_L = 2.0 * np.sin(2.0 * alpha)
        q = 0.5 * rho_air * v_a**2

        F_D_A = q * A * C_D
        F_L_A = q * A * C_L

        v_A_hat = v_A / v_a
        v_A_perp_hat = np.array([v_A_hat[1], -v_A_hat[0]])

        F_A = -F_D_A * v_A_hat + F_L_A * v_A_perp_hat
        F_A_x, F_A_y = F_A
        tau_A = d_A * (np.sin(theta) * F_A_x - np.cos(theta) * F_A_y)

    S_dot[3] = (F_A_x + F_H_x) / M
    S_dot[4] = (F_A_y + F_H_y) / M
    S_dot[5] = (tau_A + tau_H) / I
    return S_dot


# RK Method
def rk4_step(fun, t, S, dt):
    k1 = fun(t, S)
    k2 = fun(t + dt/2, S + dt*k1/2)
    k3 = fun(t + dt/2, S + dt*k2/2)
    k4 = fun(t + dt,   S + dt*k3)
    return S + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


if __name__ =="__main__":
    # Simulation settings
    t0, tf = 0, 120
    N = 500 # Step number
    dt = (tf - t0) / N

    t_values = np.linspace(t0, tf, N+1)
    S = np.zeros((6, N+1))
    S[:,0] = np.array([0.0, 0.0, np.pi/2, 0.0, 2.9, 0.0])

    # main RK4 loop
    for i in range(N):
        S[:, i+1] = rk4_step(sail_drone_dynamics, t_values[i], S[:, i], dt)

    x_trajectory = S[0, :]
    y_trajectory = S[1, :]
    x_dot = S[3, :]
    y_dot = S[4, :]

    # Plotting Trail
    plt.figure(figsize=(10, 8))
    plt.plot(x_trajectory, y_trajectory, label='Drone Trajectory')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Saildrone Trajectory')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting Velocity - Time graph
    velocity = np.sqrt(x_dot**2 + y_dot**2)

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Drone Speed vs Time")
    plt.grid(True)
    plt.show()