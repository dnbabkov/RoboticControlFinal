"""Example of setting dynamic parameters for the UR5e robot.

This script demonstrates how to modify dynamic parameters of the robot including:
- Joint damping coefficients
- Joint friction
- Link masses and inertias

The example uses a simple PD controller to show the effects of parameter changes.
"""

import numpy as np
from simulator import Simulator
from pathlib import Path
import os
from typing import Dict
import pinocchio as pin
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

q_history, dq_history, q_err_history, dq_err_history, control_history = [], [], [], [], []

def make_graphs(times, q_history, dq_history, q_err_history, dq_err_history, control_history):
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(q_history.shape[1]):
        plt.plot(times, q_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint Positions (without chatter, Phi = 80).png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(dq_history.shape[1]):
        plt.plot(times, dq_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint Velocities (without chatter, Phi = 80).png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(dq_history.shape[1]):
        plt.plot(q_history[:, i], dq_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Joint positions [rad]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Phase Portraits')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint Phase Portraits (without chatter, Phi = 80).png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(q_err_history.shape[1]):
        plt.plot(times, q_err_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Position Errors [rad]')
    plt.title('Joint Position Errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint Position Errors (without chatter, Phi = 80).png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(control_history.shape[1]):
        plt.plot(times, control_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Control [rad/s]')
    plt.title('Control over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Control (without chatter, Phi = 80).png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for i in range(q_err_history.shape[1]):
        plt.plot(times, q_err_history[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Position Errors [rad/s]')
    plt.title('Joint Position Errors over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/Joint Position Errors (without chatter, Phi = 80).png')
    plt.close()

    

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space PD controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """

    global q_history
    global dq_history
    global q_err_history
    global dq_err_history
    global control_history

    pin.computeAllTerms(model, data, q, dq)
    M = data.M
    nle = data.nle

    est_margin = 0.5

    L, _ = np.linalg.eig(M)

    # Control gains tuned for UR5e
    kp = np.array([200, 400, 160, 10, 10, 0.1])
    kd = np.array([20, 40, 40, 2, 2, 0.01])
    
    Lambda = np.diag([200, 400, 160, 10, 10, 0.1])
    k = 1000
    Phi = 80

    # Target joint configuration
    q0 = np.array([-1.4, -1.3, 1., 0, 0, 0])
    dq0 = np.zeros(6)
    ddq0 = np.zeros(6)

    q_err = q0 - q
    dq_err = dq0 - dq

    s = dq_err + Lambda @ q_err
    sigma_max = np.max(L) + est_margin
    rho = k/sigma_max*np.linalg.inv(M)

    if np.linalg.norm(s) > Phi:
        v_s = rho @ s/np.linalg.norm(s)
    else:
        v_s = rho @ s/Phi
    v = ddq0 + Lambda@dq_err + v_s

    # PD control law
    #tau = kp * (q0 - q) - kd * dq
    # Regular Inverse Dynamics
    tau = M@v + nle
    #print("Control input =", tau, "\n\nCurrent q = ", q, "\n\nCurrent q error = ", q_err)

    q_history.append(q)
    dq_history.append(dq)
    q_err_history.append(q_err)
    dq_err_history.append(dq_err)
    control_history.append(tau)

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)

    global q_history
    global dq_history
    global q_err_history
    global dq_err_history
    global control_history
    
    # Initialize simulator
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,  # Using joint space control
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/FinalVideo_Robust_Control_Without_Chattering(Phi = 80).mp4",
        fps=30,
        width=1920,
        height=1080
    )

    # Set joint damping (example values, adjust as needed)
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Set joint friction (example values, adjust as needed)
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    
    # Get original properties
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print(f"\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Add the end-effector mass and inertia
    sim.modify_body_properties(ee_name, mass=3)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")

    t0 = 0.0
    tf = 10.0
    
    # Set controller and run simulation
    sim.set_controller(joint_controller)
    sim.run(time_limit=tf)

    q_history = np.array(q_history)
    dq_history = np.array(dq_history)
    q_err_history = np.array(q_err_history)
    dq_err_history = np.array(dq_err_history)
    control_history = np.array(control_history)

    t = np.linspace(t0, tf, 5000)

    make_graphs(t, q_history, dq_history, q_err_history, dq_err_history, control_history)

if __name__ == "__main__":
    main() 