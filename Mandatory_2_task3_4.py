import numpy as np
from scipy.linalg import expm, sqrtm
import matplotlib.pyplot as plt

# np.random.seed(42)
import numpy as np
from scipy.linalg import expm, sqrtm

class ContinuousDiscreteSystem:
    def __init__(self, params):
        # Store continuous parameters
        self.F = params['F']
        self.L = params['L']
        self.G = params['G']
        self.H = np.atleast_2d(params['H'])
        self.Q_tilde = params['Q_tilde']
        self.R = params['R']
        self.P_hat_0 = params['P_hat_0']
        self.t_0 = params['t_0']
        self.t_f = params['t_f']
        self.delta_t = params['delta_t']
        self.state_dim = self.F.shape[0]
        
        # Discretize the system
        self.discretize()
        
    def discretize(self):
        # Discretize system dynamics: dx/dt = Fx + Lu
        self.La, self.Fi = self._cp2dp(self.F, self.L, self.delta_t)
        
        # Discretize process noise
        self.Ga = self._cp2dpGa(self.F, self.G, self.Q_tilde, self.delta_t)
        
        # Calculate discrete process noise covariance
        self.S = self._cp2dpS(self.F, self.G, self.Q_tilde, self.delta_t)

        # print("Discretized matrices:\n")
        # print("Fi:\n ", self.Fi, "\n")
        # print("La:\n ", self.La, "\n")
        # print("Ga:\n ", self.Ga, "\n")
        # print("S:\n ", self.S, "\n")
        
    def _cp2dp(self, F, L, d):
        #Discretize the linear system: dx/dt = Fx + Lu

        n, m = F.shape[0], L.shape[1]

        # Construct the augmented matrix [F L; 0 0]
        F_tilde = np.block([
            [F,             L],
            [np.zeros((m, n)), np.zeros((m, m))]
        ])
        M = expm(F_tilde * d)
        n = F.shape[0] # extract number of rows in F

        Fi = M[:n, :n]
        La = M[:n, n:]

        return La, Fi
    
    def _cp2dpGa(self, F, G, Q_tilde, d):
        #Discretize the process noise: Gv

        n = F.shape[0]
        GQG_T = G @ Q_tilde @ G.T  # (n x n)

        F_tilde_tilde = np.block([
            [F,               GQG_T],
            [np.zeros((n, n)), -F.T]
        ])

        M = expm(F_tilde_tilde * d)

        S = M[:n, n:]
        Phi22 = M[n:, n:]
        Q_d = S @ Phi22

        #print(f"Q_d: \n {Q_d}")

        # Cholesky factor (Gamma), due to Q_d being close to Positive semi-definite, 
        # cholesky wont work so an alterantive methos is utilized, sqrtm():
        Ga = sqrtm(Q_d).real
        return Ga
    
    def _cp2dpS(self, F, G, Q_tilde, d):
        #Calculate the discrete process noise covariance matrix S.

        n = F.shape[0]
        F_tilde_tilde = np.block([[F, G@(Q_tilde@G.T)],
                                 [np.zeros((n,n)), -F.T]])

        M = expm(F_tilde_tilde*d)
        S = M[:n, n:]

        return S

# Parameters
params = {
    'T2': 5,
    'T3': 1,
    'delta_t': 0.01,
    'R': 1,
    't_0': 0,
    't_f': 100,
    'F': np.array([[0, 1, 0],
                   [0, -1/5, 1/5],  # Using T2=5
                   [0, 0, -1/1]]),  # Using T3=1
    'L': np.array([[0],
                   [0],
                   [1/1]]),         # Using T3=1
    'G': np.array([[0],
                   [0],
                   [1]]),
    'H': np.array([1, 0, 0]),
    'Q_tilde': np.array([[2 * 0.1**2]]),
    'P_hat_0': np.diag([1, 0.1**2, 0.1**2])
}

# Plotting for task 3
def plot(x_det, x_stoch, time):
    u = np.ones_like(time)

    plt.figure(figsize=(10, 6))
    plt.plot(time, x_det[1, :], label='x2 deterministic')
    plt.plot(time, x_stoch[1, :], label='x2 stochastic')
    plt.plot(time, u, '--', label='u = 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity x2')
    plt.legend()
    plt.grid(True)
    plt.title('Velocity x2(k) with and without process noise')
    plt.show()

# Plotting for task 4 the Kalman Filter
def plot_kalman(x_hat, x_bar, x_true, P_hat, P_bar, t_0=0, t_f=100):    
    # Time vector
    N = x_hat.shape[1]
    t = np.linspace(t_0, t_f, N)
    u = np.ones_like(t)
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Index for the velocity state (second state)
    velocity_idx = 1
    
    # 1. First plot: Ground truth, prediction, and corrected velocity
    axs[0].plot(t, x_true[velocity_idx, :], 'r-', label='Ground Truth (x2)')
    axs[0].plot(t, x_bar[velocity_idx, :], 'g--', label='Predicted (x_bar2)')
    axs[0].plot(t, x_hat[velocity_idx, :], 'b-', label='Corrected (x_hat2)')
    axs[0].plot(t, u, label='Input U')
    axs[0].set_ylabel('Velocity')
    axs[0].set_title('Velocity: Ground Truth, Prediction, and Corrected Estimate')
    axs[0].grid(True)
    axs[0].legend()
    
    # 2. Second plot: Filtered velocity error with confidence bounds
    filtered_error = x_true[velocity_idx, :] - x_hat[velocity_idx, :]
    
    # Extract standard deviations for velocity from P_hat
    s_hat2 = np.zeros(N)
    for i in range(N):
        s_hat2[i] = np.sqrt(P_hat[i][velocity_idx, velocity_idx])
    
    axs[1].plot(t, filtered_error, 'b-', label='Filtered Error (x2 - x_hat2)')
    axs[1].plot(t, s_hat2, 'r--', label='+s_hat2')
    axs[1].plot(t, -s_hat2, 'r--', label='-s_hat2')
    axs[1].fill_between(t, -s_hat2, s_hat2, color='r', alpha=0.2)
    axs[1].set_ylabel('Error')
    axs[1].set_title('Filtered Velocity Error with ±s_hat2')
    axs[1].grid(True)
    axs[1].legend()
    
    # 3. Third plot: Prediction error with confidence bounds
    prediction_error = x_true[velocity_idx, :] - x_bar[velocity_idx, :]
    
    # Extract standard deviations for velocity from P_bar
    s_bar2 = np.zeros(N)
    for i in range(N):
        s_bar2[i] = np.sqrt(P_bar[i][velocity_idx, velocity_idx])
    
    axs[2].plot(t, prediction_error, 'g-', label='Prediction Error (x2 - x_bar2)')
    axs[2].plot(t, s_bar2, 'r--', label='+s_bar2')
    axs[2].plot(t, -s_bar2, 'r--', label='-s_bar2')
    axs[2].fill_between(t, -s_bar2, s_bar2, color='r', alpha=0.2)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Error')
    axs[2].set_title('Prediction Error with ±s_bar2')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

def simulation(system, meas_rate=1):
    N = int((system.t_f - system.t_0)/system.delta_t)
    time = np.linspace(system.t_0, system.t_f, N)
    state_dim = system.Fi.shape[0]
    u = 1
    
    # Calculate steps between measurements
    # meas_rate is in Hz (measurements per second)
    # delta_t is the time step in seconds
    # So (1/meas_rate) gives seconds per measurement
    # And (1/meas_rate)/delta_t gives steps per measurement
    meas_interval = int(1/(meas_rate * system.delta_t))

    # Task 3 initialization
    x_det = np.zeros((3, N))
    x_stoch = np.zeros((3, N)) # This is the ground truth for measurements for kalman filter

    # Kalman Filter Initialization
    x_bar = np.zeros((state_dim, N))
    x_hat = np.zeros((state_dim, N))
    P_bar = np.zeros((N, state_dim, state_dim))
    P_hat = np.zeros((N, state_dim, state_dim))

    # Initial conditions - generate random initial state with covariance P_hat_0
    x_hat[:,0] = np.random.multivariate_normal(np.zeros(state_dim), system.P_hat_0)
    x_bar[:,0] = x_hat[:,0]
    P_hat[0] = system.P_hat_0
    x_stoch[:, 0] = x_hat[:,0]

    # Ensure H is properly shaped as a matrix (row vector)
    H = np.atleast_2d(system.H)

    # For loop for both simulations
    for t in range(1,N):
        # Deterministic and Stochastic simulation
        x_det[:, t] = system.Fi @ x_det[:, t-1] + system.La.flatten() * u
        w = np.random.randn(system.Ga.shape[1])
        x_stoch[:, t] = system.Fi @ x_stoch[:, t-1] + system.La.flatten() * u + system.Ga @ w

        # Kalman Filter Simulation
        # Time update (prediction)
        x_bar[:,t] = system.Fi @ x_hat[:,t-1] + system.La.flatten() * u
        P_bar[t] = system.Fi @ P_hat[t-1] @ system.Fi.T + system.S
        
        # Initialize current estimate to predicted value
        x_hat[:,t] = x_bar[:,t]
        P_hat[t] = P_bar[t]

        if t % meas_interval == 0:
            # Calculate innovation covariance
            S_innovation = H @ P_bar[t] @ H.T + np.array([[system.R]])
            
            # Kalman gain
            K = P_bar[t] @ H.T @ np.linalg.inv(S_innovation)
            
            # Measurement noise
            meas_noise = np.random.normal(0, np.sqrt(system.R))
            # Measurement update
            Z = H @ x_stoch[:,t] + meas_noise
            x_hat[:,t] = x_bar[:,t] + K @ ( Z - H @ x_bar[:,t])
            P_hat[t] = (np.eye(state_dim) - K @ H) @ P_bar[t]

    return x_det, x_stoch, x_hat, x_bar, P_hat, P_bar, time

def main():

    # Create system instance
    system = ContinuousDiscreteSystem(params)
    system.discretize()
    
    x_det, x_stoch, x_hat, x_bar, P_hat, P_bar, time = simulation(system)
    
    plot(x_det, x_stoch, time)
    plot_kalman(x_hat, x_bar, x_stoch, P_hat, P_bar) 

if __name__=="__main__":
    main()

