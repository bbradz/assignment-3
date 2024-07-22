"""This script defines the main function for the assignment."""


def main():
    """Define hyperparameters and test either the SARSA or SARSA(λ) algorithm."""

    game = "Taxi-v3"  # CartPole-v0 or Taxi-v3
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.25
    lambda_value = 0.9
    num_episodes = 1500
    num_runs = 10

    learner = Learner(alpha, gamma, epsilon, lambda_value, game)

    ### Run SARSA Algorithm [uncomment the 3 lines below for testing]
    # episode_rewards_1, learned_policy = avg_episode_rewards(game, "sarsa", alpha, gamma, epsilon, lambda_value, num_episodes, num_runs)
    # plot_rewards(episode_rewards_1, game, algorithm="sarsa")
    # render_visualization(learned_policy, game, model="sarsa", algorithm="sarsa")

    ### Run SARSA-λ Algorithm [uncomment the 3 lines below for testing]
    # episode_rewards_2, learned_policy = avg_episode_rewards(game, "sarsa_lambda", alpha, gamma, epsilon, lambda_value, num_episodes, num_runs)
    # plot_rewards(episode_rewards_2, game, algorithm="sarsa_lambda")
    # render_visualization(learned_policy, game, model="sarsa_lambda", algorithm="sarsa_lambda")


if __name__ == "__main__":
    main()
