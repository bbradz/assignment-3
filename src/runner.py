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

    def run_episode(self):
        """Run a single episode using the stored environment and agent."""
        self.env.reset()


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
