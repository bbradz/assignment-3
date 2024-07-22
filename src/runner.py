"""This module defines a class to run an agent in a Gymnasium environment."""

import gymnasium as gym

from .learner import Learner


class EnvRunner:
    """A class to store and run an RL agent in a Gymnasium environment."""

    def __init__(self, env_name: str, agent: Learner):
        """Initialize the runner with an environment and learning agent.

        :param      env_name        Name of a Gymnasium environment
        """
        self.env = gym.make(env_name)  # Create and store the environment
        self.agent = agent

    def average_over_trials(self, use_lambda: bool, num_episodes: int, num_trials: int):
        """Measure average performance over a number of multi-episode trials.

        Runs the specified learning algorithm over many trials, each consisting of
            the given number of episodes, with performance averaged for each
            trial over all episodic rewards in the trial.

        :param      use_lambda      Indicates whether to run SARSA or SARSA(λ)
        :param      num_episodes    Number of episodes in each performance trial
        :param      num_trials      Number of trials to run, each with many episodes

        :returns    List of averaged rewards per episode over all the trials
                    Policy learned by the agent during the last trial
        """
        sum_rewards: list[float] = []  # Per-episode rewards summed across trials

        for _ in range(num_trials):
            self.env.reset()
            self.agent.reset()

            policy, q_values, episode_rewards = (
                self.agent.sarsa_lambda(self.env, num_episodes)
                if use_lambda
                else self.agent.sarsa(self.env, num_episodes)
            )  # TODO: Verify return types of these methods!

            if not sum_rewards:
                sum_rewards = episode_rewards
            else:
                sum_rewards = [sum(pair) for pair in zip(sum_rewards, episode_rewards)]

        average_rewards = [total / num_trials for total in sum_rewards]

        return average_rewards, policy
