import gym
import copy
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt

def discretize_state(state):
    """
    Discretizes state based on game into a single integer in the range 1-500.
    
    Input:
    state (list): A list containing the state variables [cart_position, cart_velocity, pole_angle, pole_velocity].
    
    Output:
    int: An integer representing the discretized state.
    """

    if type(state)==int:
        return state

    else:
        state_ranges = [
        (-4.8, 4.8),   # cart_position
        (-3.4, 3.4),   # cart_velocity
        (-0.418, 0.418), # pole_angle
        (-3.4, 3.4)    # pole_velocity
        ]

        bin_indices = []
        bins_per_variable = [4, 5, 5, 5]  # 500 bins total

        for value, (low, high), bins in zip(state, state_ranges, bins_per_variable):
            clipped_value = np.clip(value, low, high)
            bin_index = int((clipped_value - low) / (high - low) * (bins - 1))
            bin_indices.append(bin_index)
        
        # Map the bin indices to a single discrete value
        discrete_state = 0
        for i, bin_index in enumerate(bin_indices):
            discrete_state *= bins_per_variable[i]
            discrete_state += bin_index

        return discrete_state + 1

class ModelHolder(object):
    def __init__(self, game, alpha, gamma, epsilon, lambda_value):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_value = lambda_value

        if game=="Taxi-v3":
            self.num_states = 500
            self.num_actions = 6
        elif game=="CartPole-v0":
            self.num_states = 500
            self.num_actions = 2

        self.qtable = np.random.uniform(low=-1, high=1, size=(self.num_states, self.num_actions))
        self.policy = np.random.random_integers(self.num_actions, size=(self.num_states, 1))

    def write(self):
        "*** saving the numpy arrays for checking ***"
        np.save("qvalues", self.qtable)
        np.save("policy", self.policy)

    def select_action(self, state, testing=False):
        qmax = float("-inf")
        bestAction = None
        amaxes = []
        for action in range(self.num_actions):
            qval = self.qtable[state, action]
            if qval > qmax:
                amaxes = [action]
                qmax = qval
            elif qval == qmax:
                amaxes.append(action)
        nMaxes = len(amaxes)
        bestAction = amaxes[np.random.randint(nMaxes)]
        if not testing:
            self.policy[state] = bestAction
            roll = np.random.random()
            if roll < self.epsilon:
                return np.random.randint(self.num_actions)

        return bestAction

    def train_select_model(self, model, env, num_episodes):
        if model=="sarsa": 
            return self.sarsa(env, num_episodes)
        elif model == "sarsa-lambda": 
            return self.sarsa_lambda(env, num_episodes)

    def sarsa(self, env, num_episodes):
        """ 
        Implement standard SARSA algorithm to update qtable and learning policy.
        Input: all parameters
        Output: This function returns the updated qtable, learning policy and the reward after each episode. 
        """

        rewards_each_learning_episode = []
        for _ in range(num_episodes):
            state = discretize_state(env.reset())
            action = self.select_action(state)                       # original [s, a]
            episodic_reward = 0
            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)       # get s' and r
                next_state = discretize_state(next_state)       
                next_action = self.select_action(next_state)         # get a'
          
                td_error = reward + self.gamma * np.max(self.qtable[next_state]) - self.qtable[state][action]
                self.qtable[state][action] += self.alpha * td_error  # update qtable by adding alpha * td * corresponding_etable

                episodic_reward += reward                            # update total reward
                state = next_state                                   # update [s, a]
                action = next_action

                if done: break

            rewards_each_learning_episode.append(episodic_reward)

        np.save(f"results/{self.game}/sarsa/qvalues", self.qtable)
        np.save(f"results/{self.game}/sarsa/policy", self.policy)
        return self.policy, self.qtable, rewards_each_learning_episode

    def sarsa_lambda(self, env, num_episodes):
        """ 
        Implement Sarsa-lambda algorithm to update qtable, etable and learning policy.
        Input: all parameters
        Output: This function returns the updated qtable, learning policy and the reward after each episode. 
        """

        rewards_each_learning_episode = []
        for _ in range(num_episodes):
            state = discretize_state(env.reset())
            action = self.select_action(state)                        # original [s, a]
            episodic_reward = 0
            done = False
            self.etable = np.zeros((self.num_states, self.num_actions))
            while not done:
                next_state, reward, done, _ = env.step(action)     # get s' and r
                next_state = discretize_state(next_state)  
                next_action = self.select_action(next_state)          # get a'
          
                self.etable *= self.lambda_value*self.gamma
                                                                      # multiply etable by gamma * lambda
                self.etable[state][action] = 1                        # change [s,a] in etable to 1
                td_error = reward + self.gamma*self.qtable[next_state][next_action] - self.qtable[state][action] 
                self.qtable += self.alpha*td_error*self.etable        # update qtable by adding alpha * td * corresponding_etable

                episodic_reward += reward                             # update total reward
                state = next_state                                    # update [s, a]
                action = next_action

                if done: break

            rewards_each_learning_episode.append(episodic_reward)

        np.save(f"results/{self.game}/sarsa-lambda/qvalues", self.qtable)
        np.save(f"results/{self.game}/sarsa-lambda/policy", self.policy)
        return self.policy, self.qtable, rewards_each_learning_episode


def avg_episode_rewards(num_runs, model, game, alpha, gamma, epsilon, lambda_value, num_episodes):
    """
    Runs the algorithm a number of times and averages the episodic rewards
    from all runs for each episode
    Input: num_runs: the number of times to run the algorithm
    Output:
        episode_rewards: a list of averaged rewards per episode over a num_runs number of times
        learned_policy: the policy learned by the last run of qLearningLearner.learn_policy() to be
        used in problem visualization
    """

    episode_rewards = []
    for _ in range(num_runs):
        env = gym.make(game)
        env.reset()
        modelSelector = \
            ModelHolder(
            game, alpha, gamma, epsilon, lambda_value)
        learned_policy, q_values, single_run_er = \
            modelSelector.train_select_model(
            model, env, num_episodes)

        # on the first iteration of this loop, episodeRewards will be empty
        if not episode_rewards: episode_rewards = single_run_er
        # add this run's ERs to previous runs in order to calculate the average later
        else: episode_rewards = [episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))]

    # Get the average over ten runs
    plt.plot([er / num_runs for er in episode_rewards])
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig(f"results/{game}/{model}/rewards_plot.png")

    env = gym.make(game, render_mode="human")
    state = discretize_state(env.reset())
    env.render()
    i = 0
    while True:
        action = learned_policy[state, 0]
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state) 
        env.render()
        print(f"Step {i}, took action {action} Reward={reward}")
        state = next_state
        i += 1
        if done:
            break

avg_episode_rewards(
    num_runs=10, 
    model="sarsa",  # sarsa or sarsa-lambda
    game="CartPole-v0", # CartPole-v0 or Taxi-v3
    alpha=0.2, 
    gamma=0.95, 
    epsilon=0.1, 
    lambda_value=0.9,
    num_episodes=25_000
)

# def test_policy():
#     policy = np.load("policy_taxi_q_learning_grading.npy")
#     env = gym.make("Taxi-v3", render_mode="human")
#     env.reset()
#     num_episodes = 10
#     rewards_each_test_episode = []
#     steps_each_test_episode = []

#     for _ in range(num_episodes):
#         state = env.reset()
#         episodic_reward = 0
#         steps = 0
#         while True:
#             action = policy[state]
#             next_state, reward, done, info = env.step(action)
#             steps += 1
            
#             episodic_reward += reward
#             state = next_state

#             if done or steps >= 200:
#                 break

#         rewards_each_test_episode.append(episodic_reward)
#         steps_each_test_episode.append(steps)

#     print("Rewards each test episode: {}".format(rewards_each_test_episode))
#     print("Steps each test episode: {}".format(steps_each_test_episode))
