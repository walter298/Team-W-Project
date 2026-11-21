import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

# Height and width
height = 1000
width = 1500

# 1. Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Kalman Filter Simulation")
font = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()

#slider = Slider(screen, 100, 150, 600, 30, min=0, max=100, step=1, initial=50)
#output = TextBox(screen, 375, 220, 50, 50, fontSize=28)

# # True hidden state (position and velocity)
x_true = np.array([300, 100])

# Kalman Filter Initialization
dt = 1/60
error_std = 30
x_est = np.array([295, 100]) # Our initial estimate
F = np.array([[1, dt],  # State transition matrix
              [0, 0.99]])
P_est = np.array([[10, 0],   # Process covariance matrix
                  [0, 1000]])
H = np.array([1, 0])         # Map hidden state to observation
q = 1e5
Q = q * np.array([[0.25 * dt**4, 0.5 * dt**3],       # Process noise covariance matrix
              [0.5 * dt**3, dt**2]])
R = error_std ** 2


# Estimation Error Initialization
kalman_error = [] # Error of kalman filter estimates
naive_error = [] # Error of single measurements
ma_error = [] # Error of moving average estimates
reg_error = [] # Error of linear regression estimates

# Store data and observations
ground_truth = []
observations = []
kalman_preds = []
naive_preds = []
ma_preds = []
reg_preds = []

g_ma_n = 15 # We consider the last n timesteps for moving average
g_reg_n = 15 # We consider the last n timesteps for regression

last_recorded_time = pygame.time.get_ticks()

# 2. Main game loop
running = True
steps = 0

while running: 
    events = pygame.event.get()
    # Handle events (such as clicking the window close button)
    for event in events:
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with white (RGB)
    screen.fill((50, 50, 50))
    #output.setText(slider.getValue())
 
    # Update the hidden state
    _, mouse_y = pygame.mouse.get_pos()
    #x_true = F @ x_true
    x_true[0] = mouse_y

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
    ground_truth.append(x_true) 
    observations.append(z)
    
    # Moving Average
    ma_n = min(g_ma_n, len(observations)-1)
    last_n = np.array(observations[-ma_n:])
    average_velocity = np.mean(last_n - np.array(observations[-(ma_n+1):-1]))
    ma_pred = np.mean(last_n) + ((ma_n-1)/2)*average_velocity
    ma_error.append(x_true - ma_pred)
    
    # Linear Regression
    reg_n = min(g_reg_n, len(observations))
    last_n_reg = np.array(observations[-reg_n:])
    X = np.arange(1, reg_n+1).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, last_n_reg) 
    reg_pred = model.predict([[reg_n]])[0]
            
    #Preds
    kalman_preds.append(x_est[0])
    naive_preds.append(z)
    ma_preds.append(ma_pred)
    reg_preds.append(reg_pred)
              
    # Error
    naive_error.append(x_true[0] - z)
    ma_error.append(x_true[0] - ma_pred)
    reg_error.append(x_true[0] - reg_pred)

    pygame.draw.circle(screen, (0, 0, 255), (500, x_true[0]), 20) # Draw ground truth
    pygame.draw.circle(screen, (255, 255, 255), (490, z), 5)      # Draw  noise
    pygame.draw.circle(screen, (255, 0, 0), (480, x_est[0]), 5)  # Draw estimate
    pygame.draw.circle(screen, (0, 255, 0), (470, ma_pred), 5)  # Draw moving average estimate
    pygame.draw.circle(screen, (0, 255, 255), (460, reg_pred), 10)  # Draw regression estimate        

    # Draw the fps
    fps_text = font.render(f"FPS: {clock.get_fps():.0f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))
    
    pygame_widgets.update(events)
    pygame.display.update()
    steps += 1
    
    clock.tick(60)
    
# 3. Clean up and exit
pygame.quit()


x_plt = np.arange(1, len(ground_truth)+1)
plt.scatter(x_plt, ground_truth, c = (0, 0, 1), label = "Ground Truth")
plt.scatter(x_plt, kalman_preds, c = (1, 0, 0), label = "Kalman Prediction")
plt.scatter(x_plt, ma_preds, c = (0, 1, 0), label = "Moving Average Prediction")
plt.scatter(x_plt, reg_preds, c = (0, 1, 1), label = "Regression Prediction")

#plt.legend(loc='outside upper left')

plt.show()