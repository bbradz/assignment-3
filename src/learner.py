"""This module defines a class representing a SARSA/SARSA(λ)-based learning agent."""

import numpy as np
import gymnasium as gym
from discretize import discretize_state

class Learner:
    """A learning-based agent implementing the SARSA and SARSA(λ) algorithms."""

    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon: float,
        lambda_: float,
        game: str = "CartPole-v0",
    ):
        """Initialize the learning agent for either the Cart Pole or Taxi game.

        :param      alpha       Learning rate during SARSA Q-value updates
        :param      gamma       Discount factor of the MDP environment
        :param      epsilon     Probability of taking a random action (epsilon-greedy)
        :param      lambda_     Scalar of n-step returns during SARSA(λ)
        :param      game        Name of the game the agent will play
        """

        self.game = game

        if game == "CartPole-v0":
            self.num_states = 750
            self.num_actions = 2
        elif game == "Taxi-v3":
            self.num_states = 500
            self.num_actions = 6
        else:
            assert False, f"Game name must be 'CartPole-v0' or 'Taxi-v3', was {game}"

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_  # Note: Python doesn't allow variables named "lambda"

        self.state = 0
        self.action = 0

        self.rng = np.random.default_rng()

        # Q-table - |S| x |A| array representing state-action values
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Policy - |S| x 1 vector representing the chosen action in each state
        self.policy = self.rng.integers(self.num_actions, size=self.num_states)

        # Eligibility trace used to accelerate SARSA learning; |S| x |A| array
        self.e_table = np.zeros((self.num_states, self.num_actions))

    def calculate_action(self, state: int) -> int:
        """Calculate the best action in the given state using the Q-table.

        Note: If multiple actions have the same Q-value, picks one randomly.

        :param          state       State represented as a Q-table index

        :returns        Best action according to the Q-table
        """
        q_max = -np.inf
        best_actions = []

        for action in range(self.num_actions):
            q_val = self.q_table[state, action]
            if q_val > q_max:
                best_actions = [action]  # Reset the list of best actions
                q_max = q_val  # Store the new best Q-value
            elif q_val == q_max:
                best_actions.append(action)  # Save any equal-value actions

        best_action = self.rng.choice(best_actions)
        self.policy[state] = best_action  # Update agent's policy for this state
        return best_action

    def sarsa(self, env: gym.Env, num_episodes: int):
        """FILL IN"""
        # rewards_each_learning_episode = []

        # for _ in #### FILL IN ####

            # state, info = env.reset()

            #### FILL IN ####
            
            # episodic_reward = 0
            # done = False

            # while not #### FILL IN ####

                #### FILL IN ####
                # episodic_reward += reward

                #### FILL IN ####

        #         if done:
        #             break

        #     rewards_each_learning_episode.append(episodic_reward)

        # np.save(f"results/{self.game}/SARSA/qvalues", self.q_table)
        # np.save(f"results/{self.game}/SARSA/policy", self.policy)
        # return self.policy, self.q_table, rewards_each_learning_episode


    def sarsa_lambda(self, env: gym.Env, num_episodes: int):
        """FILL IN"""
        # rewards_each_learning_episode = []

        # for _ in #### FILL IN ####

            # episodic_reward = 0
            # done = False
            # state, info = env.reset()

            #### FILL IN ####

            # while not #### FILL IN ####:

                #### FILL IN ####

                # episodic_reward += reward

                #### FILL IN ####

        #         if done: break 

        #     rewards_each_learning_episode.append(episodic_reward)

        # np.save(f"results/{self.game}/SARSA-Lambda/qvalues", self.q_table)
        # np.save(f"results/{self.game}/SARSA-Lambda/policy", self.policy)
        # return self.policy, self.q_table, rewards_each_learning_episode