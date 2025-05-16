import numpy as np
import matplotlib.pyplot as plt
from Mandatory_2_task3_4 import params, ContinuousDiscreteSystem

def simulation(system, meas_rate=1, simulate_dynamics=True):
    N = int((system.t_f - system.t_0)/system.delta_t)
    time = np.linspace(system.t_0, system.t_f, N)
    state_dim = system.Fi.shape[0]
    u = 1
    meas_interval = int(1/(meas_rate * system.delta_t))

    # Full system simulation arrays
    if simulate_dynamics:
        x_det = np.zeros((state_dim, N))
        x_stoch = np.zeros((state_dim, N))
    else:
        x_det = None
        x_stoch = None

    x_bar = np.zeros((state_dim, N))
    x_hat = np.zeros((state_dim, N))
    P_bar = np.zeros((N, state_dim, state_dim))
    P_hat = np.zeros((N, state_dim, state_dim))

    x_hat[:, 0] = np.random.multivariate_normal(np.zeros(state_dim), system.P_hat_0)
    x_bar[:, 0] = x_hat[:, 0]
    P_hat[0] = system.P_hat_0
    if simulate_dynamics:
        x_stoch[:, 0] = x_hat[:, 0]

    H = np.atleast_2d(system.H)

    for t in range(1, N):
        if simulate_dynamics:
            x_det[:, t] = system.Fi @ x_det[:, t-1] + system.La.flatten() * u
            w = np.random.randn(system.Ga.shape[1])
            x_stoch[:, t] = system.Fi @ x_stoch[:, t-1] + system.La.flatten() * u + system.Ga @ w

        # Kalman filter prediction
        x_bar[:, t] = system.Fi @ x_hat[:, t-1] + system.La.flatten() * u
        P_bar[t] = system.Fi @ P_hat[t-1] @ system.Fi.T + system.S
        x_hat[:, t] = x_bar[:, t]
        P_hat[t] = P_bar[t]

        if t % meas_interval == 0:
            S_innovation = H @ P_bar[t] @ H.T + np.array([[system.R]])
            K = P_bar[t] @ H.T @ np.linalg.inv(S_innovation)

            meas_noise = np.random.normal(0, np.sqrt(system.R))
            Z = (H @ x_stoch[:, t] if simulate_dynamics else H @ x_bar[:, t]) + meas_noise
            x_hat[:, t] = x_bar[:, t] + K @ (Z - H @ x_bar[:, t])
            P_hat[t] = (np.eye(state_dim) - K @ H) @ P_bar[t]

    return x_det, x_stoch, x_hat, x_bar, P_hat, P_bar, time


params_filter = {
    'T2': 5,
    'T3': 1,
    'delta_t': 0.01,
    'R': 1,
    't_0': 0,
    't_f': 100,
    'F': np.array([[0, 1],
                   [0, -1/5]]), 
    'L': np.array([[0],
                   [1/5]]),   
    'G': np.array([[0],
                   [1/5]]),  # Used for process noise
    'H': np.array([1, 0]),
    'Q_tilde': np.array([[2 * 0.1**2]]),
    'P_hat_0': np.diag([1, 0.1**2])
}


sim_model = ContinuousDiscreteSystem(params)
system_filter = ContinuousDiscreteSystem(params_filter)


x_det, x_stoch_true, x_hat_opt, x_bar, P_hat_opt, P_bar, time = simulation(sim_model) # kalman filter simualtion of original system 
_, _, x_hat_sub, _, P_hat_sub, _, _ = simulation(system_filter) # Simualtion of filter model


N = np.array([[1, 0, 0],
              [0, 1, 0]])

x_hat_sub_full = np.array([N.T @ x_hat_sub[:, i] for i in range(x_hat_sub.shape[1])]).T  # Re-map the reduced state vector back to 3D
error = x_stoch_true - x_hat_sub_full # The error of the sub optimal comapred to the true state

std_error = np.std(error, axis=1)  # empirical std per state
std_opt = np.sqrt(P_hat_opt[:, [0,1], [0,1]].T)  # for position and velocity
std_sub = np.sqrt(P_hat_sub[:, [0,1], [0,1]].T)  # for reduced model


fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- 1. Position std deviation ---
axs[0].plot(time, std_opt[0], label='Optimal KF', linestyle='-')
axs[0].plot(time, std_sub[0], label='Suboptimal KF', linestyle='--')
axs[0].hlines(std_error[0], xmin=time[0], xmax=time[-1], label='Empirical error', colors='black', linestyle=':')
axs[0].set_ylabel('Std Dev (Position)')
axs[0].legend()
axs[0].grid(True)

# --- 2. Velocity std deviation ---
axs[1].plot(time, std_opt[1], label='Optimal KF', linestyle='-')
axs[1].plot(time, std_sub[1], label='Suboptimal KF', linestyle='--')
axs[1].hlines(std_error[1], xmin=time[0], xmax=time[-1], label='Empirical error', colors='black', linestyle=':')
axs[1].set_ylabel('Std Dev (Velocity)')
axs[1].legend()
axs[1].grid(True)

# --- 3. Velocity comparison ---
axs[2].plot(time, x_stoch_true[1], label='True velocity', linestyle='-')
axs[2].plot(time, x_hat_opt[1], label='Optimal KF estimate', linestyle='--')
axs[2].plot(time, x_hat_sub_full[1], label='Suboptimal KF estimate', linestyle=':')
axs[2].set_ylabel('Velocity')
axs[2].set_xlabel('Time [s]')
axs[2].legend()
axs[2].grid(True)

plt.suptitle("Suboptimal vs Optimal Kalman Filter Performance", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()