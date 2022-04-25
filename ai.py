import numpy as np
import random

# Interesting link !
# https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/
# http://www.angusgriffith.com/2019/12/31/beating-minesweeper-with-neural-networks.html

class AI:
    def __init__(self, learning_rate=0.9, probab_init=0.5):

        self.boardsize = 0
        self.proba = None
        self.leara = learning_rate
        self.payoffs = None # Matrix avec proba d'être une bombe
        self.epsilon = 0.75 #0 = random | 1 = exploit

    def setboardsize(self, size):
        self.boardsize = size

    def setPayoffs(self, payoff_mat):
        self.payoffs = payoff_mat
        #print(self.payoffs)
        #input()

    def updatePayoffs(self, payoff_mat):
        for i in range(0, self.boardsize-1):
            for j in range(0, self.boardsize-1):
                if(self.payoffs[i,j]==0):
                    self.payoffs[i,j] = payoff_mat[i,j]

    def setProba(self, size):
        self.proba = np.zeros((size,size))

    def learn(self, stimu, x, y):
        if (stimu >= 0):
            newprob = self.proba[x,y] + (1 - self.proba[x,y]) * self.leara * stimu
        else:
            newprob = self.proba[x,y] + self.proba[x,y] * self.leara * stimu
        self.proba[x,y] = newprob
        #self.proba[1 - x,y] = 1 - newprob
        return self.proba

    def act(self):
        # IDEE DE BASE
        # Use trained model to predict all tiles in the perimeter of the uncover tiles

        # if (no tiles with low prob of bomb in perimeter):
        #   choose random tiles
        #   x = random.randint(0, boardsize) # assurer que x, y pas déjà uncover
        #   y = random.randint(0, boardsize)
        # else:
        #   choose tiles with lowest prob of being of bomb
        #return x, y

        # NOUVELLE IDEE AVEC RL
        #print(self.proba)
        #return np.random.choice([0, 1, 2, 3], p=self.proba)
        #
        if(random.uniform(0, 1) < self.epsilon):  # Exploit
            return self.getMaxIndex()
            print("Exploit")
        else:  # Explore
            #print("Explore")
            return random.randint(0, self.boardsize-1),random.randint(0, self.boardsize-1)

    # Get max index of tiles in the perimeter
    def getMaxIndex(self):
        tmp_max = -1
        max_x = 0
        max_y = 0
        for i in range(0,len(self.proba)):
            for j in range(0,len(self.proba)):
                if(self.proba[i, j]>tmp_max):
                    tmp_max = self.proba[i, j]
                    max_x = i
                    max_y = j
        return max_x, max_y

    def play(self):
        max_x, max_y = self.getMaxIndex()
        #self.proba[max_x, max_y] = 0
        #print(max_x, max_y)
        return max_x, max_y

    def predict(self):
        pass
