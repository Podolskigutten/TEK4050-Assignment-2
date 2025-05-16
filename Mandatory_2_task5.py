from Mandatory_2_task3_4 import ContinuousDiscreteSystem, params, simulation
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_sim(N, M, system):
    Ø = np.zeros((N, M)) # each element is the error at time K for the simulation J, for K = 1,2,3,...,M and J = 1,2,3,...,N, and error is x - x_hat 
    Æ = np.zeros((N, M))
    x2_bar_MC = np.zeros((N,M))
    x2_hat_MC = np.zeros((N,M))
    s2_k = np.zeros(M)

    for j in range(N):
        _, x_true, x_hat, x_bar, P_hat, _, _ = simulation(system)
        
        s2_k += np.sqrt(P_hat[:, 1, 1])  # shape: (M,), Diagonal element corresponding to velocity    
        x2_bar_MC[j,:] = x_bar[1,:]
        x2_hat_MC[j,:] = x_hat[1,:]
        Ø[j, :] = x_true[1, :] - x_hat[1, :]
        Æ[j, :] = x_true[1, :] - x_bar[1, :]

    s2_k /= N
    mean = np.mean(Ø, axis=0)              # m_hat
    covariance = np.var(Ø, axis=0, ddof=1) # P_hat
    std_dev = np.sqrt(covariance)          # s_hat

    return Ø, Æ, x2_bar_MC, x2_hat_MC, mean, std_dev, s2_k

def plot1(x_bar_MC, x_hat_MC):
    """Plot Kalman filter simulations with paired x_bar and x_hat estimates."""
    plt.figure(figsize=(10, 6))
    
    # Get the number of time steps from the data
    n_steps = len(x_bar_MC[0])
    
    # Create time array from 0 to 100 seconds with 0.01 second steps
    time = np.linspace(0, 100, n_steps)
    
    for i in range(len(x_bar_MC)):
        color = plt.cm.tab10(i % 10)
        plt.plot(time, x_bar_MC[i], '--', color=color)
        plt.plot(time, x_hat_MC[i], '-', color=color)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('State Estimate')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot2(Ø, Æ):
    """Plot individual errors for each simulation"""
    plt.figure(figsize=(10, 6))
    
    # Get the number of time steps from the data
    N, M = Ø.shape
    n_steps = M
    
    # Create time array from 0 to 100 seconds (assuming same time range as plot1)
    time = np.linspace(0, 100, n_steps)
    
    for i in range(N):
        color = plt.cm.tab10(i % 10)
        # Plot xi_2(k) - x̂i_2(k) as solid line
        plt.plot(time, Ø[i, :], '-', color=color, label=f'$x_2 - \hat{{x}}_2$ (sim {i+1})' if i == 0 else "")
        # Plot xi_2(k) - x̄i_2(k) as dashed line
        plt.plot(time, Æ[i, :], '--', color=color, label=f'$x_2 - \overline{{x}}_2$ (sim {i+1})' if i == 0 else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Error')
    plt.title('Estimation Errors for Each Simulation')
    plt.grid(alpha=0.3)
    
    # Add a legend that shows the line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='-'),
                    Line2D([0], [0], color='black', linestyle='--')]
    plt.legend(custom_lines, ['$x_2 - \hat{x}_2$', '$x_2 - \overline{x}_2$'], 
               loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot3_4(mean_error, sample_std, kalman_std):
    """
    Plot the mean error, sample standard deviation, and Kalman standard deviation
    for the velocity component (x_2) over time.
    """
    n_steps = len(mean_error)
    time = np.linspace(0, 100, n_steps)

    plt.figure(figsize=(10, 6))
    
    plt.plot(time, mean_error, label=r'$\hat{m}_2^N(k)$ (mean error)', linewidth=2)
    plt.plot(time, sample_std, label=r'$\hat{s}_2^N(k)$ (sample std)', linewidth=2)
    plt.plot(time, kalman_std, label=r'$\hat{s}_2(k)$ (Kalman std)', linewidth=2)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Value')
    plt.title('Velocity Error Statistics')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # create instance of sytem to simulate
    system = ContinuousDiscreteSystem(params)

    # Allocate memory for trajectory error
    M = int((params['t_f'] - params['t_0'])/params['delta_t']) # How many iteration
    N = 10 # Number of simulations

    Ø, Æ, x2_bar_MC, x2_hat_MC, mean, std_dev, s2_k = monte_carlo_sim(N, M, system)
    
    plot1(x2_bar_MC, x2_hat_MC) # Plot 1
    plot2(Ø, Æ) #Plot 2
    plot3_4(mean, std_dev, s2_k) # Plot 3

    # Plot 4 requires 100 simualtions
    N = 100
    Ø, Æ, x2_bar_MC, x2_hat_MC, mean, std_dev, s2_k = monte_carlo_sim(N, M, system)

    plot3_4(mean, std_dev, s2_k) # Plot 4

    return 0

if __name__ == "__main__":
    main()
