import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import pandas as pd

random.seed(1738)

# Main loop
def run_simulation(reg_n_, error_std_, a, b, show_plots):
    
    running = True
    steps = 0

    max_steps = 300
    
    g_reg_n = reg_n_[a]
    error_std = error_std_[b]

    # # True hidden state (position and velocity)
    x_true = np.array([300, 100])

    # Estimation Error Initialization
    kalman_error = [] # Error of kalman filter estimates
    naive_error = [] # Error of single measurements
    reg_error = [] # Error of linear regression estimates

    # Store data and observations
    ground_truth = []
    observations = []
    kalman_preds = []
    naive_preds = []
    reg_preds = []
    
    # Kalman Filter Initialization
    dt = 1/60
    x_est = np.array([295, 100]) # Our initial estimate
    F = np.array([[1, dt],  # State transition matrix
              [0, 1]])
    P_est = np.array([[10, 0],   # Process covariance matrix
                  [0, 1000]])
    H = np.array([1, 0])         # Map hidden state to observation
    q = 1e5
    Q = q * np.array([[0.25 * dt**4, 0.5 * dt**3],       # Process noise covariance matrix
              [0.5 * dt**3, dt**2]])
    R = error_std ** 2

    while running: 
        if steps >= max_steps:
            running = False
 
        # Update the hidden state
        x_true = F @ x_true

        # Predict Step: Update estimate and process covariance
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q

        # Simulate a noisy observation, using the error_std standard deviation
        z = random.gauss(mu=x_true[0], sigma=error_std)

        # Update Step:
        K = P_pred @ H.T / (H @ P_pred @ H.T + R)     # Kalman Gain
        x_est = x_pred + K * (z - H @ x_pred)         # Hidden state estimate
        P_est = (np.eye(2) - np.outer(K, H)) @ P_pred # Process covariance estimate

        # Record
        ground_truth.append(x_true[0]) 
        observations.append(z)
    
        # Linear Regression
        reg_n = min(g_reg_n, len(observations))
        last_n_reg = np.array(observations[-reg_n:])
        X = np.arange(1, reg_n+1).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, last_n_reg) 
        reg_pred = model.predict([[reg_n]])[0]
            
        #Preds
        naive_preds.append(z)
        kalman_preds.append(x_est[0])
        reg_preds.append(reg_pred)
              
        # Error
        naive_error.append(x_true[0] - z)
        kalman_error.append(x_true[0] - x_est[0])
        reg_error.append(x_true[0] - reg_pred)

        steps += 1
        
    v_naive_mse = np.sum(np.square(np.array(naive_error[-(len(naive_error)-g_reg_n):])))/(len(naive_error)-g_reg_n)
    v_kalman_mse = np.sum(np.square(np.array(kalman_error[-(len(kalman_error)-g_reg_n):])))/(len(kalman_error)-g_reg_n)
    v_reg_mse = np.sum(np.square(np.array(reg_error[-(len(reg_error)-g_reg_n):])))/(len(reg_error)-g_reg_n)
        
    if(show_plots):
        ## Plot 1
        size = 3
        x_plt = np.arange(1, len(ground_truth)+1)
        plt.scatter(x_plt, ground_truth, c = (0, 0, 1), label = "Ground Truth", s = size)
        plt.scatter(x_plt, naive_preds, color = "black", label = "Noisy Observation", s = size)
        plt.scatter(x_plt, reg_preds, c = (0, 1, 1), label = "Regression Prediction", s = size)
        plt.scatter(x_plt, kalman_preds, c = (1, 0, 0), label = "Kalman Prediction", s = size)

        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.subplots_adjust(right=0.75)
        plt.suptitle("Plot of Predictions and Actual vs Time")
        plt.title(f'For last_n: {g_reg_n} and Error Std: {error_std}')
        plt.xlabel('Timestep')
        plt.ylabel('Position (m)')

        plt.show()

        ## Plot 2
        size = 3
        plt.scatter(x_plt, x_plt - x_plt, c=(0, 0, 1), s = size)
        plt.scatter(x_plt, naive_error, color = "black", label = "Observation Error", s = size)
        plt.scatter(x_plt, reg_error, c = (0, 1, 1), label = "Regression Error", s = size)
        plt.scatter(x_plt, kalman_error, c = (1, 0, 0), label = "Kalman Error", s = size)

        plt.suptitle("Plot of Error vs Time")
        plt.title(f'For last_n: {g_reg_n} and Error Std: {error_std}')
        plt.xlabel('Timestep')
        plt.ylabel('Error (m)')
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.subplots_adjust(right=0.75)

        plt.show()

        #Print Error
        print("Naive Error: ", v_naive_mse)
        print("Kalman Error: ", v_kalman_mse)
        print("Regression Error: ", v_reg_mse)
    
    error_array = [g_reg_n, error_std, v_naive_mse, v_kalman_mse, v_reg_mse]
    return(error_array)


# vars
g_reg_n_arr = [5, 10, 30]
error_std_arr = [1, 5, 20]

error_list = []

# Reg Loop
for i in range(len(g_reg_n_arr)):
    for j in range(len(error_std_arr)):
        error_list.append(run_simulation(g_reg_n_arr, error_std_arr, i, j, False))
        

error_df = pd.DataFrame(error_list, columns=['Last_n', 'Error Std', 'Observational Error', 'Kalman Error', 'Regression Error'])

print(error_df)



run_simulation(g_reg_n_arr, error_std_arr, 1, 2, True)


error_df.to_csv("error.csv", index=False)
