import numpy as np
from numpy import random,maximum


class QAgent:
    def __init__(self, learning_rate: float, epsilon_max: float = None, epsilon_min: float = None, epsilon_decay: float = None):
        """
        :param num_actions: Number of actions.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.

        no gamma because immediate reward
        """
        self.q_table = []
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, observation: list):
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        index = self.get_index(observation)
        max_value = max(self.q_table[index][1:])
        action = self.q_table[index].index(max_value,0,3)
        return action

    def act(self, observation: list, training: bool):
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        if training:
            #state not in Q-table when training -> add in Q-table
            if not any(observation in x for x in self.q_table):
                self.q_table.append([observation,0,0])
            exp_prob = random.random()
            if exp_prob < self.epsilon: #explore
                action = random.randint(1,3)
                return action
        #ignore undiscovered states when NOT training
        elif not any(observation in x for x in self.q_table):
            print("NOT IN Q-TABLE")
            return -1
        #greedy action
        return self.greedy_action(observation)

    def learn(self, obs: list, act: int, rew: float, done: bool):
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        """
        index = self.get_index(obs)
        self.q_table[index][act] = self.q_table[index][act] + self.learning_rate * (rew - self.q_table[index][act])
        if done:    #update exploration rate if episode done
            self.epsilon = maximum(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get_index(self, obs: list):
        """
        Find index of the observation in the Q-table

        :param obs: The observation.
        :return: The index of the observation in the Q-table.
        """
        for i,x in enumerate(self.q_table):
            if obs in x:
                return i
