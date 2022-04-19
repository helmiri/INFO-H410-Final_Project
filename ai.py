import numpy as np
import random

# Interesting link !
#https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/

class AI:
    def __init__(self):
        self.boardsize = 0

    def setboardsize(self, size):
        self.boardsize = size

    def learn(self):
        pass

    def play(self):
        x= 0; y = 0
        # Use trained model to predict all tiles in the perimeter of the uncover tiles

        # if (no tiles with low prob of bomb in perimeter):
        #   choose random tiles
        #   x = random.randint(0, boardsize) # assurer que x, y pas déjà uncover
        #   y = random.randint(0, boardsize)
        # else:
        #   choose tiles with lowest prob of being of bomb
        return x, y

    def predict(self):
        pass
