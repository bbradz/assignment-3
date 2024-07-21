import gym
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo

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


def avg_episode_rewards(
    game, algorithm, alpha, gamma, epsilon, lambda_value, num_episodes, num_runs
):
    """
    Runs the learner algorithms a number of times and averages the episodic rewards
    from all runs for each episode

    Input: num_runs: the number of times to run the learner
           algorithm: the algorithm to use ("sarsa" or "sarsa_lambda")

    Output:
        episode_rewards: a list of averaged rewards per episode over a num_runs number of times
        learned_policy: the policy learned by the last run of the learner
    """
    episode_rewards = []
    for _ in range(num_runs):
        env = gym.make(game)
        env.reset()
        learner = Learner(alpha, gamma, epsilon, lambda_value, game)
        if algorithm == "sarsa":
            learned_policy, q_values, single_run_er = learner.learn_policy_sarsa(
                env, num_episodes
            )
        elif algorithm == "sarsa_lambda":
            learned_policy, q_values, single_run_er = learner.learn_policy_sarsa_lambda(
                env, num_episodes
            )

        if not episode_rewards:
            episode_rewards = single_run_er
        else:
            episode_rewards = [
                episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))
            ]

    episode_rewards = [er / num_runs for er in episode_rewards]
    return episode_rewards, learned_policy


game = "Taxi-v3"  # CartPole-v0 or Taxi-v3
alpha = 0.2
gamma = 0.9
epsilon = 0.25
lambda_value = 0.9
num_episodes = 1500
num_runs = 10

### Run SARSA Algorithm [uncomment the 3 lines below for testing]
# episode_rewards_1, learned_policy = avg_episode_rewards(game, "sarsa", alpha, gamma, epsilon, lambda_value, num_episodes, num_runs)
# plot_rewards(episode_rewards_1, game, algorithm="sarsa")
# render_visualization(learned_policy, game, model="sarsa", algorithm="sarsa")

### Run SARSA-λ Algorithm [uncomment the 3 lines below for testing]
# episode_rewards_2, learned_policy = avg_episode_rewards(game, "sarsa_lambda", alpha, gamma, epsilon, lambda_value, num_episodes, num_runs)
# plot_rewards(episode_rewards_2, game, algorithm="sarsa_lambda")
# render_visualization(learned_policy, game, model="sarsa_lambda", algorithm="sarsa_lambda")
