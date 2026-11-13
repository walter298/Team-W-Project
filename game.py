import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import random

# 1. Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("Kalman Filter Simulation")
clock = pygame.time.Clock()

slider = Slider(screen, 100, 150, 600, 30, min=0, max=100, step=1, initial=50)
output = TextBox(screen, 375, 220, 50, 50, fontSize=28)

# True path
#xTrue = 

# Kalman Filter Initialization
delta_t = 0.01
x = np.array([50, 50])     # Hidden state (position and velocity)
F = np.array([[1, delta_t], # State transition matrix
              [0, 1]])

last_recorded_time = pygame.time.get_ticks()
ball_height = 700

# 2. Main game loop
running = True
while running:
    events = pygame.event.get()
    # Handle events (such as clicking the window close button)
    for event in events:
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with white (RGB)
    screen.fill((0, 255, 255))
    output.setText(slider.getValue())
    
    if ball_height > 1000:
        ball_height = 700
        
    # Draw a blue circle at the center (x=500, y=500) with radius 50
    pygame.draw.circle(screen, (0, 0, 255), (500, x[0]), 50)
    
    if pygame.time.get_ticks() - last_recorded_time > delta_t / 1000:
        x = F @ x
        error = random.gauss(mu=x[0], sigma=10)
        pygame.draw.circle(screen, (), (), 10)
        last_recorded_time = pygame.time.get_ticks()
    
    pygame_widgets.update(events)
    pygame.display.update()
    
    clock.tick(60)
    
# 3. Clean up and exit
pygame.quit()