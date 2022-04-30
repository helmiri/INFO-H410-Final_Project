import numpy as np
import random


# Interesting link !
# https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/
# http://www.angusgriffith.com/2019/12/31/beating-minesweeper-with-neural-networks.html

class AI:
    def __init__(self):
        self.boardsize = 0
        self.positions_revealed = []

    def setboardsize(self, size):
        self.boardsize = size

    def reset(self):
        self.positions_revealed = []

    def act(self, probmine, peri, current_revealed):
        # Get prob minimum of mine in the perimeter
        return self.getMinProbPeri(peri, probmine, current_revealed)


    """
    Return the position of the tile in the perimeter with the minimum probability of being a mine
    """
    def getMinProbPeri(self, peri, probmine, positions_revealed):
        prob_to_evaluate = []
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed):
                prob_to_evaluate.append(probmine[0][pos[0], pos[1]])
        minval = np.amin(prob_to_evaluate)
        index = np.where(prob_to_evaluate == np.amin(prob_to_evaluate))

        return peri[index[0][0]][0], peri[index[0][0]][1]


    """
    Return the position of the tile in the perimeter with the maximum probability of being a mine
    """
    def getMaxProbPeri(self, peri, probmine):
        global positions_revealed
        tmp_max = 0
        max_i = None
        max_j = None
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed and probmine[pos[0], pos[1]] > tmp_max):
                tmp_max = probmine[pos[0], pos[1]]
                max_i = pos[0]
                max_j = pos[1]
        #probmine[min_i, min_j]= 1000000
        #positions_revealed.append((max_i,max_j))
        return max_i, max_j

    """
    Return the position of the tile on the board with the maximum probability of being a mine
    """
    def getMinProbMine(self, probmine):
        tmp_min = 10000000
        min_i = None
        min_j = None
        for i in range(0, len(probmine)):
            for j in range(0, len(probmine)):
                if(probmine[i,j] < tmp_min):
                    tmp_min = probmine[i,j]
                    min_i = i
                    min_j = j
        return min_i, min_j
