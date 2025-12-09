"""Integrated Saildrone Simulation + Pygame GUI

This file contains a full runnable Python script that:
- runs the saildrone simulation (wind gusts, current, actuator limits, roll/pitch)
- stores the simulation states
- opens a Pygame GUI to visualise the boat, sail, rudder, wind/current vectors, trail
- provides simple playback controls: Space=play/pause, Right/Left=change speed, S=step, Q/ESC=quit

Place this file in the same folder as `saildrone_hydro.dat` and run with Python 3.
"""

import sys
import os
import math
import numpy as np
import base64
import matplotlib.pyplot as plt

# ---- Load hydrodynamics data module (same mechanism you provided) ----
FILENAME = 'saildrone_hydro.dat'
if not os.path.exists(FILENAME):
    raise FileNotFoundError(f"Required data file '{FILENAME}' not found in working directory.")
_data = open(FILENAME, "r").read()
exec(base64.b64decode(_data).decode('utf-8'))

def get_vals(velocity, heading, turn_rate, rudder_angle):
    force, torque = _eval_data(velocity, heading, turn_rate, rudder_angle)
    return force, torque

# ---------------------------
# Physical & model constants
# ---------------------------
M = 2500.0
I_yaw = 10000.0
A = 15.0
d_A = 0.1
rho_air = 1.225

# roll/pitch parameters (tuneable)
I_roll = 2000.0
I_pitch = 3000.0
K_roll = 1.0e4
D_roll = 1.0e3
K_pitch = 1.0e4
D_pitch = 1.0e3
h_sail = 2.0
h_pitch = 1.0

# Base wind vector (Easterly)
v_W_base = np.array([-6.7, 0.0])

# ---------------------------
# Environmental models
# ---------------------------
def wind_model(t):
    gust = 0.10 * math.sin(0.2 * t)
    max_sweep_deg = 5.0
    sweep_angle = math.radians(max_sweep_deg) * math.sin(0.01 * t)
    R = np.array([[math.cos(sweep_angle), -math.sin(sweep_angle)],
                  [math.sin(sweep_angle),  math.cos(sweep_angle)]])
    return (1.0 + gust) * (R @ v_W_base)

def current_model(t):
    u = 0.25 + 0.05 * math.sin(0.03 * t)
    v = 0.05 * math.sin(0.02 * t + 0.5)
    return np.array([u, v])

# ---------------------------
# Controller (commanded angles)
# ---------------------------
def get_control_inputs(t):
    if 0 <= t < 60:
        return math.radians(-45.0), math.radians(0.0)
    elif 60 <= t < 65:
        return math.radians(-22.5), math.radians(2.1)
    else:
        return math.radians(-22.5), math.radians(0.0)

# ---------------------------
# Actuator rate limiter
# ---------------------------
def rate_limit(target, current, max_rate, dt):
    diff = target - current
    limit = max_rate * dt
    diff = np.clip(diff, -limit, limit)
    return current + diff

# ---------------------------
# Dynamics function (returns S_dot and updated actuator angles)
# State S (12): x,y,theta,x_dot,y_dot,theta_dot,beta_s,beta_r,phi,phi_dot,gamma,gamma_dot
# ---------------------------
def sail_drone_dynamics(t, S, dt):
    x, y, theta, x_dot, y_dot, theta_dot, beta_s, beta_r, phi, phi_dot, gamma, gamma_dot = S
    S_dot = np.zeros_like(S)

    # Kinematics
    S_dot[0] = x_dot
    S_dot[1] = y_dot
    S_dot[2] = theta_dot

    # Actuator command + limits
    beta_s_cmd, beta_r_cmd = get_control_inputs(t)
    BETA_S_MAX = math.radians(180)
    BETA_R_MAX = math.radians(30)
    MAX_S_RATE = math.radians(15)
    MAX_R_RATE = math.radians(10)

    beta_s_new = rate_limit(beta_s_cmd, beta_s, MAX_S_RATE, dt)
    beta_r_new = rate_limit(beta_r_cmd, beta_r, MAX_R_RATE, dt)
    beta_s_new = np.clip(beta_s_new, -BETA_S_MAX, BETA_S_MAX)
    beta_r_new = np.clip(beta_r_new, -BETA_R_MAX, BETA_R_MAX)

    S_dot[6] = (beta_s_new - beta_s) / dt
    S_dot[7] = (beta_r_new - beta_r) / dt

    # Environment
    v_W = wind_model(t)
    v_current = current_model(t)
    v_D = np.array([x_dot, y_dot])

    # Hydrodynamics (relative to water)
    v_water_rel = v_D - v_current
    F_H, tau_H = get_vals(v_water_rel, theta, theta_dot, beta_r_new)
    F_H_x, F_H_y = F_H

    # Aerodynamics
    v_A = v_W - v_D
    v_a = np.linalg.norm(v_A)
    F_A_x = F_A_y = tau_A = 0.0
    M_roll_aero = 0.0
    M_pitch_aero = 0.0

    if v_a > 1e-6:
        phi_A = math.atan2(v_A[1], v_A[0])
        phi_sail = theta + beta_s_new
        alpha = math.atan2(math.sin(phi_sail - phi_A), math.cos(phi_sail - phi_A))

        C_D = 1.0 - math.cos(2.0 * alpha)
        C_L = 2.0 * math.sin(2.0 * alpha)
        q = 0.5 * rho_air * v_a**2
        F_D_A = q * A * C_D
        F_L_A = q * A * C_L

        v_A_hat = v_A / v_a
        v_A_perp_hat = np.array([v_A_hat[1], -v_A_hat[0]])

        # Simple 3D coupling: reduce lift/drag by roll/pitch
        lift_reduction = max(0.0, math.cos(phi))
        drag_reduction = max(0.0, math.cos(gamma))

        F_A_x = -F_D_A * drag_reduction * v_A_hat[0] + F_L_A * lift_reduction * v_A_perp_hat[0]
        F_A_y = -F_D_A * drag_reduction * v_A_hat[1] + F_L_A * lift_reduction * v_A_perp_hat[1]

        tau_A = d_A * (math.sin(theta) * F_A_x - math.cos(theta) * F_A_y)

        M_roll_aero = F_L_A * lift_reduction * h_sail * np.sign(math.cos(alpha))
        M_pitch_aero = F_D_A * drag_reduction * h_pitch * np.sign(math.cos(alpha))

    # Translational accelerations and yaw
    S_dot[3] = (F_A_x + F_H_x) / M
    S_dot[4] = (F_A_y + F_H_y) / M
    S_dot[5] = (tau_A + tau_H) / I_yaw

    # Roll
    phi_ddot = (M_roll_aero - K_roll * phi - D_roll * phi_dot) / I_roll
    S_dot[8] = phi_dot
    S_dot[9] = phi_ddot

    # Pitch
    gamma_ddot = (M_pitch_aero - K_pitch * gamma - D_pitch * gamma_dot) / I_pitch
    S_dot[10] = gamma_dot
    S_dot[11] = gamma_ddot

    return S_dot, beta_s_new, beta_r_new

# ---------------------------
# RK4 integrator
# ---------------------------
def rk4_step(fun, t, S, dt):
    k1, b1_s, b1_r = fun(t, S, dt)
    k2, _, _ = fun(t + dt/2.0, S + dt*k1/2.0, dt)
    k3, _, _ = fun(t + dt/2.0, S + dt*k2/2.0, dt)
    k4, _, _ = fun(t + dt, S + dt*k3, dt)
    S_new = S + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
    S_new[6] = b1_s
    S_new[7] = b1_r
    return S_new

# ---------------------------
# Precompute simulation states
# ---------------------------
def run_simulation(t0=0.0, tf=120.0, N=1000):
    dt = (tf - t0) / N
    t_values = np.linspace(t0, tf, N+1)
    S = np.zeros((12, N+1))
    S[:,0] = np.array([0.0, 0.0, math.pi/2.0, 0.0, 2.9, 0.0,
                       math.radians(-45.0), 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(N):
        S[:, i+1] = rk4_step(sail_drone_dynamics, t_values[i], S[:, i], dt)
    return t_values, S

# ---------------------------
# Pygame GUI
# ---------------------------

# ---------------------------
# Standalone vector drawing helper (no nested defs)
# ---------------------------
def draw_vector(screen, vec, color, start_pos):
    import pygame, math
    start = start_pos
    end = (int(start[0] + vec[0]*30), int(start[1] - vec[1]*30))
    pygame.draw.line(screen, color, start, end, 3)
    ang = math.atan2(-(end[1]-start[1]), end[0]-start[0])
    ah = 8
    p1 = (end[0] - ah*math.cos(ang-math.pi/6), end[1] + ah*math.sin(ang-math.pi/6))
    p2 = (end[0] - ah*math.cos(ang+math.pi/6), end[1] + ah*math.sin(ang+math.pi/6))
    if (p1 == p2) or (p1 == end) or (p2 == end):
        p1 = (end[0]-5, end[1]-3)
        p2 = (end[0]-5, end[1]+3)
    pygame.draw.polygon(screen, color, [end, p1, p2])
    return start, end

# ---------------------------
# GUI launcher (clean structure, no nested functions)
# ---------------------------
def launch_gui(t_values, S, speed=1.0):
    import pygame, math
    pygame.init()

    W, H = 1000, 800
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Saildrone Simulation')
    clock = pygame.time.Clock()

    scale = 4.0  # zoom factor
    origin = np.array([W//2, H//2])
    font = pygame.font.SysFont('Arial', 16)

    index = 0
    playback_rate = speed
    paused = False
    trail = []
    max_trail = 500
    running = True

    def world_to_screen(pt):
        return (int(origin[0] + pt[0]*scale), int(origin[1] - pt[1]*scale))

    while running:
        # ---- Events ----
        # Camera pan parameters
        pan_speed = 40

        # Handle zoom separately (mouse wheel)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                # zoom in/out
                if event.y > 0:
                    scale *= 1.1
                else:
                    scale /= 1.1
                scale = max(0.5, min(scale, 40))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    playback_rate = min(playback_rate * 2.0, 16.0)
                elif event.key == pygame.K_LEFT:
                    playback_rate = max(playback_rate / 2.0, 0.125)
                elif event.key == pygame.K_s:
                    paused = True
                    index = min(index + 1, len(t_values) - 1)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        # Camera panning via arrow keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            origin[1] += pan_speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            origin[1] -= pan_speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            origin[0] += pan_speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            origin[0] -= pan_speed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    playback_rate = min(playback_rate * 2.0, 16.0)
                elif event.key == pygame.K_LEFT:
                    playback_rate = max(playback_rate / 2.0, 0.125)
                elif event.key == pygame.K_s:
                    paused = True
                    index = min(index + 1, len(t_values) - 1)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if not paused:
            index = int(min(index + playback_rate, len(t_values)-1))

        # ---- Clear ----
        screen.fill((12, 24, 40))

        # ---- State extraction ----
        x, y, theta = S[0, index], S[1, index], S[2, index]
        vx, vy = S[3, index], S[4, index]
        beta_s, beta_r = S[6, index], S[7, index]
        phi = S[8, index]
        gamma = S[10, index]
        t = t_values[index]

        # ---- Draw trail ----
        trail.append((x, y))
        if len(trail) > max_trail:
            trail.pop(0)
        if len(trail) > 1:
            pts = [world_to_screen(p) for p in trail]
            pygame.draw.lines(screen, (180,220,180), False, pts, 2)

        # ---- Draw wind/current arrows ----
        vW = wind_model(t)
        vC = current_model(t)
        draw_vector(screen, vW, (255,200,50), (80, H-80))
        draw_vector(screen, vC, (100,200,255), (160, H-80))
        screen.blit(font.render("Wind", True, (255,200,50)), (80, H-60))
        screen.blit(font.render("Current", True, (100,200,255)), (160, H-60))

        # ---- Draw boat ----
        hull = np.array([[4,0], [-3,1.2], [-3,-1.2]])
        hull[:,1] *= math.cos(phi)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        hull_world = (R @ hull.T).T + np.array([x,y])
        hull_screen = [world_to_screen(p) for p in hull_world]
        pygame.draw.polygon(screen, (220,220,240), hull_screen)
        pygame.draw.polygon(screen, (60,90,120), hull_screen, 2)

        # ---- Sail ----
        sail = np.array([[0.5,0], [3.5,0]])
        R_sail = np.array([[math.cos(beta_s), -math.sin(beta_s)],
                           [math.sin(beta_s),  math.cos(beta_s)]])
        sail_world = (R @ (R_sail @ sail.T)).T + np.array([x,y])
        sail_screen = [world_to_screen(p) for p in sail_world]
        pygame.draw.line(screen, (240,240,180), sail_screen[0], sail_screen[1], 4)

        # ---- Rudder ----
        rud = np.array([[-3.1,0], [-4.1,0]])
        R_r = np.array([[math.cos(beta_r), -math.sin(beta_r)],
                         [math.sin(beta_r),  math.cos(beta_r)]])
        rud_world = (R @ (R_r @ rud.T)).T + np.array([x,y])
        rud_screen = [world_to_screen(p) for p in rud_world]
        pygame.draw.line(screen, (200,120,120), rud_screen[0], rud_screen[1], 3)

        # ---- HUD ----
        info = [
            f"t = {t:.1f} s",
            f"speed = {math.hypot(vx,vy):.2f} m/s",
            f"heading = {math.degrees(theta)%360:.1f} deg",
            f"sail = {math.degrees(beta_s):.1f} deg",
            f"rudder = {math.degrees(beta_r):.1f} deg",
            f"roll = {math.degrees(phi):.2f} deg",
            f"pitch = {math.degrees(gamma):.2f} deg",
            f"playback rate = {playback_rate}x",
        ]
        y0 = 10
        for line in info:
            screen.blit(font.render(line, True, (200,200,200)), (W-250, y0))
            y0 += 20

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ---------------------------
# Main entry
# ---------------------------
if __name__ == '__main__':
    print('Running simulation (this may take a few seconds)...')
    t_vals, S = run_simulation(t0=0.0, tf=120.0, N=1000)
    print('Simulation complete. Launching GUI...')
    launch_gui(t_vals, S, speed=1.0)
    print('Done.')
