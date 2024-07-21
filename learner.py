class Learner:
    def __init__(
        self,
        alpha,
        gamma,
        epsilon,
        lambda_value,
        game: Literal["CartPole-v0", "Taxi-v3"] = "CartPole-v0",
    ):
        self.game = game
        if game == "CartPole-v0":
            self.num_states = 500
            self.num_actions = 2
        elif game == "Taxi-v3":
            self.num_states = 500
            self.num_actions = 6
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_value = lambda_value
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(
            low=-1, high=1, size=(self.num_states, self.num_actions)
        )
        self.etable = np.zeros((self.num_states, self.num_actions))
        self.policy = np.random.random_integers(
            self.num_actions, size=(self.num_states, 1)
        )

    def write(self):
        "*** saving the numpy arrays for checking ***"
        np.save("qvalues", self.qtable)
        np.save("policy", self.policy)

    def calculate_action(self, state, testing=False):
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
        self.policy[state] = bestAction
        return bestAction

    def learn_policy_sarsa_lambda(self, env, num_episodes):
        """FILL IN"""

        # rewards_each_learning_episode = []

        # for ### FILL IN ###

        # episodic_reward = 0
        # done = False

        ### FILL IN ###

        # while ### FILL IN ###

        ### FILL IN ###

        # episodic_reward += reward

        ### FILL IN ###

        # if done: break

        # rewards_each_learning_episode.append(episodic_reward)

        # np.save(f"results/{self.game}/sarsa_lambda/qvalues", self.qtable)
        # np.save(f"results/{self.game}/sarsa_lambda/policy", self.policy)
        # return self.policy, self.qtable, rewards_each_learning_episode

    def learn_policy_sarsa(self, env, num_episodes):
        """FILL IN"""

        # rewards_each_learning_episode = []

        # for ### FILL IN ###

        # episodic_reward = 0
        # done = False

        # while ### FILL IN ###:

        ### FILL IN ###

        # episodic_reward += reward

        ### FILL IN ###

        # if done:
        #    break

        # rewards_each_learning_episode.append(episodic_reward)

        # np.save(f"results/{self.game}/sarsa/qvalues", self.qtable)
        # np.save(f"results/{self.game}/sarsa/policy", self.policy)
        # return self.policy, self.qtable, rewards_each_learning_episode
