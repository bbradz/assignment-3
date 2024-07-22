import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from .learner import Learner


def discretize_state(state):
    """
    Discretizes state based on game into a single integer in the range 1-500.

    Input:
    state (list): A list containing the state variables [cart_position, cart_velocity, pole_angle, pole_velocity].

    Output:
    int: An integer representing the discretized state.
    """

    if type(state) == int:
        return state

    else:
        state_ranges = [
            (-4.8, 4.8),  # cart_position
            (-3.4, 3.4),  # cart_velocity
            (-0.418, 0.418),  # pole_angle
            (-3.4, 3.4),  # pole_velocity
        ]

        bin_indices = []
        bins_per_variable = [4, 5, 5, 5]  # 500 bins total

        for value, (low, high), bins in zip(state, state_ranges, bins_per_variable):
            clipped_value = np.clip(value, low, high)
            bin_index = int((clipped_value - low) / (high - low) * (bins - 1))
            bin_indices.append(bin_index)

        discrete_state = 0
        for i, bin_index in enumerate(bin_indices):
            discrete_state *= bins_per_variable[i]
            discrete_state += bin_index

        return discrete_state + 1


def plot_rewards(episode_rewards, game, algorithm):
    """
    Plots a learning curve for the specified algorithm
    Input: episode_rewards: a list of episode rewards
    """
    plt.plot(episode_rewards)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig(f"results/{game}/{algorithm}/rewards_plot_{algorithm}.png")
    plt.close()


def render_visualization(learned_policy, game, model):
    """
    Renders a taxi problem visualization
    Input: learned_policy: the learned policy to be used by the taxi
    """
    env = gym.make(game, render_mode="rgb_array")
    env = RecordVideo(
        env, f"results/{game}/{model}", episode_trigger=lambda episode_id: True
    )
    state = discretize_state(env.reset())
    env.render()
    while True:
        action = learned_policy[state, 0]
        next_state, reward, done, info = env.step(action)
        next_state = discretize_state(next_state)
        env.render()
        print(f"Took action {action} Reward={reward}")
        state = next_state
        if done:
            break
