import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize Pygame
pygame.init()

# Set up the game window
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game with GAN and RNN")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake properties
snake_block = 20
snake_speed = 15

# GAN model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()

# RNN model for playing Snake
class SnakeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Initialize the RNN
input_size = 6  # [snake_x, snake_y, food_x, food_y, direction_x, direction_y]
hidden_size = 128
output_size = 4  # [left, right, up, down]
rnn_model = SnakeRNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Game functions
def generate_food():
    noise = torch.randn(1, 100)
    with torch.no_grad():
        generated = generator(noise).numpy()[0]
    x = int(generated[0] * (width - snake_block))
    y = int(generated[1] * (height - snake_block))
    return x, y

def draw_snake(snake_list):
    for block in snake_list:
        pygame.draw.rect(window, GREEN, [block[0], block[1], snake_block, snake_block])

def get_state(snake_list, food):
    head = snake_list[-1]
    if len(snake_list) > 1:
        direction = [
            (snake_list[-1][0] - snake_list[-2][0]) / snake_block,
            (snake_list[-1][1] - snake_list[-2][1]) / snake_block
        ]
    else:
        direction = [0, 0]  # No movement at the start
    
    return [
        head[0] / width,
        head[1] / height,
        food[0] / width,
        food[1] / height,
        direction[0],  # x direction
        direction[1],  # y direction
    ]

def get_action(state, hidden):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        q_values, new_hidden = rnn_model(state_tensor, hidden)
        return torch.argmax(q_values).item(), new_hidden

def game_loop():
    global window  # Make window a global variable

    game_over = False
    game_close = False

    x1 = width / 2
    y1 = height / 2

    x1_change = 0
    y1_change = 0

    snake_list = [[x1, y1]]
    length_of_snake = 1

    foodx, foody = generate_food()

    clock = pygame.time.Clock()
    hidden = rnn_model.init_hidden()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
                pygame.quit()
                return length_of_snake - 1

        if game_close:
            window.fill(BLACK)
            font = pygame.font.SysFont(None, 50)
            message = font.render(f"Game Over! Score: {length_of_snake - 1}", True, WHITE)
            window.blit(message, [width / 4, height / 3])
            pygame.display.update()

            pygame.time.wait(2000)
            return length_of_snake - 1

        state = get_state(snake_list, (foodx, foody))
        action, hidden = get_action(state, hidden)

        if action == 0:  # Left
            x1_change = -snake_block
            y1_change = 0
        elif action == 1:  # Right
            x1_change = snake_block
            y1_change = 0
        elif action == 2:  # Up
            y1_change = -snake_block
            x1_change = 0
        elif action == 3:  # Down
            y1_change = snake_block
            x1_change = 0

        x1 += x1_change
        y1 += y1_change

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_close = True

        window.fill(BLACK)
        pygame.draw.rect(window, RED, [foodx, foody, snake_block, snake_block])
        snake_head = [x1, y1]
        snake_list.append(snake_head)

        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for block in snake_list[:-1]:
            if block == snake_head:
                game_close = True

        draw_snake(snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx, foody = generate_food()
            length_of_snake += 1

        clock.tick(snake_speed)

    return length_of_snake - 1

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    # Reinitialize Pygame window for each episode
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake Game with RNN")

    score = game_loop()
    print(f"Episode {episode + 1}/{num_episodes}, Score: {score}")

    # Ensure Pygame is quit after each episode
    pygame.quit()

print("Training complete")
