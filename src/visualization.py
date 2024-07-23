"""This module defines functions to visualize and plot the agent's performance."""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

from discretize import discretize_state


def plot_rewards(episode_rewards: list[float], env_name: str, algorithm: str):
    """Plot a learning curve for the specified algorithm.

    :param      episode_rewards     List of per-episode rewards, averaged across trials
    :param      env_name            Name of the agent's environment
    :param      algorithm           Name of the learning algorithm used
    """
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Averaged reward per episode")
    plt.ion()
    plt.savefig(f"results/{env_name}/{algorithm}/rewards_plot_{algorithm}.png")
    plt.close()


def render_visualization(learned_policy: np.ndarray, env_name: str, algorithm: str):
    """Render an episode of the given learned policy in the specified environment.

    :param      learned_policy      Policy represented as an |S| x 1 vector
    :param      env_name            Name of the environment to be used
    :param      algorithm           Name of the learning algorithm used
    """
    output_path = f"results/{env_name}/{algorithm}"

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, output_path)

    state, _ = env.reset()
    state_idx = discretize_state(state)
    env.render()
    while True:
        action = learned_policy[state_idx]
        next_state, reward, done, truncated, _ = env.step(action)
        state_idx = discretize_state(next_state)
        env.render()
        print(f"Took action: {action} Reward: {reward}")
        state = next_state
        if done:
            break
