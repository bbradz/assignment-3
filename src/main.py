"""This script defines the main function for the assignment."""

from .learner import Learner
from .runner import EnvRunner
from .visualization import plot_rewards, render_visualization


def main():
    """Define hyperparameters and test either the SARSA or SARSA(λ) algorithm."""

    game = "Taxi-v3"  # CartPole-v0 or Taxi-v3
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.25
    lambda_ = 0.9

    n_episodes = 1500
    n_runs = 10

    agent = Learner(alpha, gamma, epsilon, lambda_, game)
    runner = EnvRunner(game, agent)

    ### Run the SARSA algorithm [uncomment the 3 lines below for testing]
    rewards1, policy1 = runner.average_over_trials(False, n_episodes, n_runs)
    plot_rewards(rewards1, game, algorithm="SARSA")
    render_visualization(policy1, game, algorithm="SARSA")

    ### Run the SARSA(λ) algorithm [uncomment the 3 lines below for testing]
    rewards2, policy2 = runner.average_over_trials(True, n_episodes, n_runs)
    plot_rewards(rewards2, game, algorithm="SARSA-Lambda")
    render_visualization(policy2, game, algorithm="SARSA-Lambda")


if __name__ == "__main__":
    main()
