import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


class FireRescueEnv(gym.Env):
    """
      0 = empty
      1 = fire
      2 = injured
      3 = fire + injured
    """
    def __init__(self, max_steps=12):
        super(FireRescueEnv, self).__init__()

        self.grid_size = 3

        self.num_single_actions = 6
        self.action_space = spaces.Discrete(self.num_single_actions * self.num_single_actions)

        low = np.zeros((13,), dtype=np.float32)
        high = np.array([2, 2, 2, 2] + [3] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_steps = max_steps
        self.current_step = 0

        self.reset()

    def _generate_random_grid(self):
        grid = np.zeros((3, 3), dtype=int)

        for r in range(3):
            for c in range(3):
                if (r, c) == (2, 0):
                    grid[r, c] = 0
                else:
                    val = random.random()
                    if val < 0.5:
                        grid[r, c] = 0
                    elif val < 0.7:
                        grid[r, c] = 1
                    elif val < 0.9:
                        grid[r, c] = 2
                    else:
                        grid[r, c] = 3
        return grid

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.grid = self._generate_random_grid()
        self.fire_pos = [2, 0]
        self.rescue_pos = [2, 0]

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        obs = [
            self.fire_pos[0],
            self.fire_pos[1],
            self.rescue_pos[0],
            self.rescue_pos[1]
        ]
        obs.extend(self.grid.flatten().tolist())
        return np.array(obs, dtype=np.float32)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def step(self, action):
        """
          0 = UP
          1 = DOWN
          2 = LEFT
          3 = RIGHT
          4 = STAY
          5 = ACT (put out fire or rescue)
        """
        self.current_step += 1

        fire_action = action // self.num_single_actions
        rescue_action = action % self.num_single_actions

        new_fire_pos = self.fire_pos.copy()
        new_rescue_pos = self.rescue_pos.copy()


        if fire_action == 0:  # up
            new_fire_pos[0] = max(0, new_fire_pos[0] - 1)
        elif fire_action == 1:  # down
            new_fire_pos[0] = min(self.grid_size - 1, new_fire_pos[0] + 1)
        elif fire_action == 2:  # left
            new_fire_pos[1] = max(0, new_fire_pos[1] - 1)
        elif fire_action == 3:  # right
            new_fire_pos[1] = min(self.grid_size - 1, new_fire_pos[1] + 1)
        elif fire_action == 4:
            pass
        elif fire_action == 5:
            pass


        if rescue_action == 0:  # up
            new_rescue_pos[0] = max(0, new_rescue_pos[0] - 1)
        elif rescue_action == 1:
            new_rescue_pos[0] = min(self.grid_size - 1, new_rescue_pos[0] + 1)
        elif rescue_action == 2:
            new_rescue_pos[1] = max(0, new_rescue_pos[1] - 1)
        elif rescue_action == 3:
            new_rescue_pos[1] = min(self.grid_size - 1, new_rescue_pos[1] + 1)
        elif rescue_action == 4:
            pass
        elif rescue_action == 5:
            pass

        if self._manhattan_distance(new_fire_pos, new_rescue_pos) > 2:
            new_fire_pos = self.fire_pos
            new_rescue_pos = self.rescue_pos

        self.fire_pos = new_fire_pos
        self.rescue_pos = new_rescue_pos

        reward = 0
        terminated = False
        truncated = False

        if fire_action == 5:
            cell_val = self.grid[self.fire_pos[0], self.fire_pos[1]]
            if cell_val in [1, 3]:
                reward += 1.0
                if cell_val == 3:
                    self.grid[self.fire_pos[0], self.fire_pos[1]] = 2
                else:
                    self.grid[self.fire_pos[0], self.fire_pos[1]] = 0

        if rescue_action == 5:
            cell_val = self.grid[self.rescue_pos[0], self.rescue_pos[1]]

            if cell_val == 2:
                reward += 1.0
                self.grid[self.rescue_pos[0], self.rescue_pos[1]] = 0


        if not np.any(self.grid == 1) and not np.any(self.grid == 2) and not np.any(self.grid == 3):
            terminated = True
            reward += 5.0

        if self.current_step >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        grid_copy = np.array(self.grid, dtype=object)
        fr, fc = self.fire_pos
        rr, rc = self.rescue_pos

        # Mark the helicopters on a copy
        grid_copy[fr, fc] = "F"
        grid_copy[rr, rc] = "R" if grid_copy[rr, rc] != "F" else "FR"

        print("\nStep:", self.current_step)
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                row_str += f"{grid_copy[r, c]} "
            print(row_str)
        print()


def evaluate_policy(env, model, num_episodes=3, render=False):

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        if render:
            print(f"----- Evaluation Episode {ep + 1} -----")
            env.render()

        while not done:
            # Model predicts action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            if render:
                env.render()
        if render:
            print(f"Episode {ep + 1} finished in {step_count} steps.\n")

env = FireRescueEnv(max_steps=12)
env_test = FireRescueEnv(max_steps=12)


print("Behavior BEFORE training (random policy):")
for i in range(2):
    obs = env_test.reset()
done = False
env_test.render()
while not done:
    action = env_test.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env_test.step(action)
    done = terminated or truncated
    env_test.render()
print("Episode finished.\n")


model = DQN(
    policy=MlpPolicy,
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.95,
    train_freq=4,  # train every 4 steps
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_final_eps=0.01,
    tensorboard_log=None
)

model.learn(total_timesteps=200*1000)

print("\nBehavior AFTER training (DQN policy):")
evaluate_policy(env_test, model, num_episodes=1, render=True)