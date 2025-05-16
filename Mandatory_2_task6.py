from Mandatory_2_task3_4 import ContinuousDiscreteSystem, params, simulation
import numpy as np
import matplotlib.pyplot as plt
import copy

def error_budget_analysis(system):
    """
    Compute individual error contributions to the total Kalman filter error
    """
    N = int((system.t_f - system.t_0)/system.delta_t)
    time = np.linspace(system.t_0, system.t_f, N)
    
    # Initialize covariance matrices for each error source
    P_P0 = system.P_hat_0.copy()  # Initial state uncertainty
    P_Q = np.zeros_like(system.P_hat_0)  # Process noise contribution
    P_R = np.zeros_like(system.P_hat_0)  # Measurement noise contribution
    
    # Storage for results
    std_P0_all = np.zeros((N, 3))
    std_Q_all = np.zeros((N, 3))
    std_R_all = np.zeros((N, 3))
    
    meas_interval = int(1/(1.0 * system.delta_t))  # 1 Hz measurements
    
    for k in range(N):
        # Store current standard deviations
        std_P0_all[k] = np.sqrt(np.diag(P_P0))
        std_Q_all[k] = np.sqrt(np.diag(P_Q))
        std_R_all[k] = np.sqrt(np.diag(P_R))
        
        if k < N-1:
            # Continuous covariance propagation
            Q_continuous = system.G @ system.Q_tilde @ system.G.T
            
            # Propagate each error source separately
            P_P0 = propagate_covariance_continuous(P_P0, system.F, 
                                                 np.zeros_like(Q_continuous), 
                                                 system.delta_t)
            P_Q = propagate_covariance_continuous(P_Q, system.F, 
                                                Q_continuous, 
                                                system.delta_t)
            P_R = propagate_covariance_continuous(P_R, system.F, 
                                                np.zeros_like(Q_continuous), 
                                                system.delta_t)
            
            # Measurement update every meas_interval steps
            if (k+1) % meas_interval == 0:
                H = np.atleast_2d(system.H)
                
                # Handle scalar R case
                if np.isscalar(system.R):
                    R_matrix = np.array([[system.R]])
                else:
                    R_matrix = system.R
                
                # For each error source
                for i, P in enumerate([P_P0, P_Q, P_R]):
                    # Innovation covariance
                    S = H @ P @ H.T + R_matrix
                    
                    # Kalman gain
                    K = P @ H.T @ np.linalg.inv(S)
                    
                    # Updated covariance
                    P_temp = (np.eye(3) - K @ H) @ P
                    
                    # Add measurement noise contribution only to P_R
                    if i == 2:  # P_R case
                        P_temp += K @ R_matrix @ K.T
                    
                    # Update the covariance matrix
                    if i == 0:
                        P_P0 = P_temp
                    elif i == 1:
                        P_Q = P_temp
                    else:
                        P_R = P_temp
    
    return std_P0_all, std_Q_all, std_R_all, time

def propagate_covariance_continuous(P, F, Q, dt):
    """
    Solves: dP/dt = F*P + P*F' + Q
    Using simple forward Euler integration
    """
    P_dot = F @ P + P @ F.T + Q
    return P + P_dot * dt

def main():
    system = ContinuousDiscreteSystem(params)
    
    # Get individual error contributions using the new error budget analysis
    std_P0_all, std_Q_all, std_R_all, time = error_budget_analysis(system)
    
    # Get total error from full simulation
    _, _, _, _, P_hat, _, _ = simulation(system)
    std_total = np.sqrt(np.array([np.diag(P) for P in P_hat]))
    
    # Extract components for each state
    # Position (state 0)
    std_P0_pos = std_P0_all[:, 0]
    std_Q_pos = std_Q_all[:, 0]
    std_R_pos = std_R_all[:, 0]
    std_tot_pos = std_total[:, 0]
    
    # Velocity (state 1)
    std_P0_vel = std_P0_all[:, 1]
    std_Q_vel = std_Q_all[:, 1]
    std_R_vel = std_R_all[:, 1]
    std_tot_vel = std_total[:, 1]
    
    # Current (state 2)
    std_P0_cur = std_P0_all[:, 2]
    std_Q_cur = std_Q_all[:, 2]
    std_R_cur = std_R_all[:, 2]
    std_tot_cur = std_total[:, 2]
    
    # Create a 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error budget for position
    ax1.plot(time, std_P0_pos, label="P₀ contribution", linewidth=2)
    ax1.plot(time, std_Q_pos, label="Q contribution", linewidth=2)
    ax1.plot(time, std_R_pos, label="R contribution", linewidth=2)
    ax1.plot(time, std_tot_pos, '--', label="Total error (KF)", linewidth=2)
    ax1.set_title("Error Budget for Position")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Standard Deviation [m]")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Error budget for velocity
    ax2.plot(time, std_P0_vel, label="P₀ contribution", linewidth=2)
    ax2.plot(time, std_Q_vel, label="Q contribution", linewidth=2)
    ax2.plot(time, std_R_vel, label="R contribution", linewidth=2)
    ax2.plot(time, std_tot_vel, '--', label="Total error (KF)", linewidth=2)
    ax2.set_title("Error Budget for Velocity")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Standard Deviation [m/s]")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Error budget for armature current
    ax3.plot(time, std_P0_cur, label="P₀ contribution", linewidth=2)
    ax3.plot(time, std_Q_cur, label="Q contribution", linewidth=2)
    ax3.plot(time, std_R_cur, label="R contribution", linewidth=2)
    ax3.plot(time, std_tot_cur, '--', label="Total error (KF)", linewidth=2)
    ax3.set_title("Error Budget for Armature Current")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Standard Deviation [A]")
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: RMS Sum vs Kalman Filter Position Error
    rms_sum_pos = np.sqrt(std_P0_pos**2 + std_Q_pos**2 + std_R_pos**2)
    ax4.plot(time, rms_sum_pos, label="RMS sum of error contributions", linewidth=2)
    ax4.plot(time, std_tot_pos, '--', label="Kalman filter position std dev", linewidth=2)
    ax4.set_title("RMS Sum vs Kalman Filter Position Error")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Position Standard Deviation [m]")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some diagnostic information
    print("Final standard deviations (t=100s):")
    print(f"Position - P0: {std_P0_pos[-1]:.4f}, Q: {std_Q_pos[-1]:.4f}, R: {std_R_pos[-1]:.4f}, Total: {std_tot_pos[-1]:.4f}")
    print(f"Velocity - P0: {std_P0_vel[-1]:.4f}, Q: {std_Q_vel[-1]:.4f}, R: {std_R_vel[-1]:.4f}, Total: {std_tot_vel[-1]:.4f}")
    print(f"Current  - P0: {std_P0_cur[-1]:.4f}, Q: {std_Q_cur[-1]:.4f}, R: {std_R_cur[-1]:.4f}, Total: {std_tot_cur[-1]:.4f}")
    
    # Verify RMS sum matches total
    final_rms_pos = np.sqrt(std_P0_pos[-1]**2 + std_Q_pos[-1]**2 + std_R_pos[-1]**2)
    print(f"\nVerification for position at t=100s:")
    print(f"RMS sum: {final_rms_pos:.4f}")
    print(f"KF total: {std_tot_pos[-1]:.4f}")
    print(f"Difference: {abs(final_rms_pos - std_tot_pos[-1]):.6f}")
    
    return fig

if __name__ == '__main__':
    main()