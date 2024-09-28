# Snake-using-RNN
Testing out bellman equation on videogames

This code implements a reinforcement learning agent that learns to play the Snake game using a Deep Q-Network (DQN). It uses Pygame for the game's visual interface and PyTorch for the neural network.  

**1. Libraries and Constants:**  Imports necessary libraries and sets game constants (memory size, batch size, learning rate, block size, game speed).

**2. Pygame Setup:** Initializes Pygame, sets up the display, and defines colors.

**3. `Direction` Enum and `Point` Namedtuple:** Defines an enumeration for movement directions (RIGHT, LEFT, UP, DOWN) and a namedtuple for representing points (x, y coordinates).

**4. `SnakeGameAI` Class:**

```python
class SnakeGameAI:
    # ...
```

This class handles the game logic and rendering.

- `__init__`: Initializes game state (direction, snake body, score, food, frame iteration).
- `reset`: Resets the game state when a game ends.
- `_place_food`: Places food randomly on the grid, avoiding the snake's body.
- `play_step`: Executes a single game step based on the agent's action.
    - Handles events (like quitting the game).
    - Moves the snake.
    - Calculates the reward.
    - Checks for collisions or timeout (game over condition).
    - Updates the score if the snake eats food.
    - Updates the UI.
    - Returns the reward, game over status, and score.
- `is_collision`: Checks if the snake has collided with itself or the boundaries.
- `_update_ui`:  Draws the game elements (snake, food, score) on the screen using Pygame.
- `_move`:  Updates the snake's position based on the chosen action.  The action is represented as a one-hot vector ([1, 0, 0] for straight, [0, 1, 0] for right turn, [0, 0, 1] for left turn).

**5. `Linear_QNet` Class:**

```python
class Linear_QNet(nn.Module):
    # ...
```

This class defines the neural network that approximates the Q-function.

- `__init__`: Initializes the network with two linear layers and ReLU activation.
- `forward`: Performs the forward pass through the network.
- `save`: Saves the model's state dictionary to a file (creates the 'model' directory if it doesn't exist).

**6. `QTrainer` Class:**

```python
class QTrainer:
    # ...
```

Handles the training process.

- `__init__`: Initializes optimizer and loss function.
- `train_step`: Performs a single training step.
    - Converts input data to tensors.
    - Calculates the predicted Q-values.
    - Calculates the target Q-values using the Bellman equation.
    - Updates the model's weights using backpropagation.

**7. `Agent` Class:**

```python
class Agent:
    # ...
```

Implements the DQN agent.

- `__init__`: Initializes game count, exploration rate (epsilon), discount factor (gamma), memory buffer (deque), Q-network, and trainer.
- `get_state`: Converts game state into a feature vector for the neural network.
- `remember`: Stores experiences (state, action, reward, next state, done) in the replay memory.
- `train_long_memory`: Trains the model on a batch of experiences sampled from replay memory.
- `train_short_memory`: Trains the model on a single experience.
- `get_action`: Chooses an action based on the epsilon-greedy strategy.

**8. `plot` Function:** Plots the scores and mean scores during training.

**9. `train` Function:** The main training loop.

- Creates the game and agent instances.
- Loops through games.
    - Gets the current state.
    - Chooses an action.
    - Plays a step in the game.
    - Trains the short-term memory.
    - Stores the experience in memory.
    - If the game is over:
        - Resets the game.
        - Trains the long-term memory.
        - Updates the record score.
        - Prints and plots the scores.
