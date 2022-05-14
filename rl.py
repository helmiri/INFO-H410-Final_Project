import numpy as np
from numpy import ndarray,random,max,maximum,where


class QAgent:
    def __init__(self, num_states: int, learning_rate: float, epsilon_max: float = None, epsilon_min: float = None, epsilon_decay: float = None):
        """
        :param num_actions: Number of actions.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.

        no gamma because immediate reward
        """
        self.q_table = np.zeros((num_states,2))
        #print(self.q_table)
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, observation: int):
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        max_value = max(self.q_table[observation, :])
        action = where(self.q_table[observation, :] == max_value)
        return action[0][0]

    def act(self, observation: int, training: bool):
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        if training:
            exp_prob = random.random()
            if exp_prob < self.epsilon: #explore
                action = random.randint(0,2)
                return action
        #greedy action
        return self.greedy_action(observation)

    def learn(self, obs: int, act: int, rew: float, done: bool):
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        """
        self.q_table[obs, act] = self.q_table[obs, act] + self.learning_rate * (rew - self.q_table[obs, act])
        if done:    #update exploration rate if episode done
            self.epsilon = maximum(self.epsilon * self.epsilon_decay, self.epsilon_min)
