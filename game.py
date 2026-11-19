import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import random

# Height and width
height = 1000
width = 1500

# 1. Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Kalman Filter Simulation")
clock = pygame.time.Clock()

#slider = Slider(screen, 100, 150, 600, 30, min=0, max=100, step=1, initial=50)
#output = TextBox(screen, 375, 220, 50, 50, fontSize=28)

# True path
#xTrue = 

# Kalman Filter Initialization
delta_t = 0.01
x = np.array([300, 100])     # Hidden state (position and velocity)
F = np.array([[1, delta_t], # State transition matrix
              [0, 1]])

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
    
    if x[0] + 20 >= height or x[0] -20 <= 0:
        x[1] *= -1
        
    # Draw a blue circle at the center (x=500, y=500) with radius 20
    pygame.draw.circle(screen, (0, 0, 255), (500, x[0]), 20)
    
    if pygame.time.get_ticks() - last_recorded_time > delta_t / 1000:
        x = F @ x
        observation = random.gauss(mu=x[0], sigma=30)
        pygame.draw.circle(screen, (255, 255, 255), (500, observation), 5)
        last_recorded_time = pygame.time.get_ticks()
    
    pygame_widgets.update(events)
    pygame.display.update()
    steps += 1
    
    clock.tick(60)
    
# 3. Clean up and exit
pygame.quit()
