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

    def act(self, probmine, peri):
        # Get prob minimum of mine in the perimeter
        return self.getMinProbPeri(peri, probmine)

        # FORGOT THAT
        """
        if(random.uniform(0, 1) < self.epsilon):  # Exploit
            #print("Exploit")
            return self.getMinProbMine(probmine)
            return self.getMaxPeri(perimeter)
            # Max on the total board
            #return self.getMaxIndex()
        else:  # Explore
            #print("Explore")
            # Random tile on the perimeter
            #return perimeter[random.randint(0, len(perimeter)-1)]
            # Random tile on the board
            return random.randint(0, self.boardsize-1),random.randint(0, self.boardsize-1)
        """


    """
    Return the position of the tile in the perimeter with the minimum probability of being a mine
    """
    def getMinProbPeri(self, peri, probmine):
        tmp_min = 10000000
        min_i = None
        min_j = None
        for pos in peri:
            if((pos[0], pos[1]) not in self.positions_revealed and probmine[pos[0], pos[1]] < tmp_min):
                tmp_min = probmine[pos[0], pos[1]]
                min_i = pos[0]
                min_j = pos[1]
        self.positions_revealed.append((min_i,min_j))
        return min_i, min_j

    """
    Return the position of the tile in the perimeter with the maximum probability of being a mine
    """
    def getMaxProbPeri(self, peri, probmine):
        tmp_max = 0
        max_i = None
        max_j = None
        for pos in peri:
            if((pos[0], pos[1]) not in self.positions_revealed and probmine[pos[0], pos[1]] > tmp_max):
                tmp_max = probmine[pos[0], pos[1]]
                max_i = pos[0]
                max_j = pos[1]
        #probmine[min_i, min_j]= 1000000
        self.positions_revealed.append((max_i,max_j))
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
