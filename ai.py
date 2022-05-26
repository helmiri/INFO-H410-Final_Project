import numpy as np
import random


# Interesting link !
# https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/
# http://www.angusgriffith.com/2019/12/31/beating-minesweeper-with-neural-networks.html


# TODO : Rename class as action
# TODO : Start avec les corners de la board

class AI:
    def __init__(self):
        self.boardsize = 0
        self.positions_revealed = []
        self.positions_flaged = []

    def setboardsize(self, size):
        self.boardsize = size

    def reset(self):
        self.positions_revealed = []
        self.positions_flaged = []

    def act(self, probmine, peri, current_revealed):
        # Get prob minimum of mine in the perimeter
        (x,y) = self.getMinProbPeri(peri, probmine, current_revealed)

        if(x == None):
            (x,y) = self.getMinProbMine(probmine)
            print("global minimum")
        self.positions_revealed.append((x,y))
        return (x,y)

    def flag(self, probmine, peri, current_revealed):
        (x,y) = self.getMaxProbPeri(peri, probmine, current_revealed)
        #(x,y) = self.getMaxProbMine(probmine)
        if(x != None):
            self.positions_flaged.append((x,y))
        return (x, y)

    """
    Return the position of the tile in the perimeter with the minimum probability of being a mine
    """
    def getMinProbPeri(self, peri, probmine, positions_revealed):
        prob_to_evaluate = {}
        for pos in peri:
            #if((pos[0], pos[1]) not in positions_revealed and (pos[0], pos[1]) not in self.positions_flaged):
            if((pos[0], pos[1]) not in positions_revealed):
                prob_to_evaluate[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
        index = min(prob_to_evaluate, key=prob_to_evaluate.get)

        # TODO : test
        if(min(prob_to_evaluate.values()) > 0.3):
            return (None, None)

        return index



    """
    Return the position of the tile in the perimeter with the maximum probability of being a mine
    """
    def getMaxProbPeri(self, peri, probmine, positions_revealed):
        prob_to_evaluate = {}
        index = (None, None)
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed and (pos[0], pos[1]) not in self.positions_flaged):
            #if((pos[0], pos[1]) not in positions_revealed):
                prob_to_evaluate[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
        if(len(prob_to_evaluate)!=0):
            index = max(prob_to_evaluate, key=prob_to_evaluate.get)
        return index

    """
    Return the position of the tile on the board with the maximum probability of being a mine
    """
    #TODO : Rewrite
    def getMinProbMine(self, probmine):
        tmp_min = 10000000
        min_i = None
        min_j = None
        for i in range(0, len(probmine[0])):
            for j in range(0, len(probmine[0])):
                if(probmine[0][i,j] < tmp_min and (i,j) not in self.positions_revealed):
                    tmp_min = probmine[0][i,j]
                    min_i = i
                    min_j = j
        print(tmp_min)
        return min_i, min_j

    # TODO : rewrite
    def getMaxProbMine(self, probmine):
        tmp_max = 0
        max_i = None
        max_j = None
        for i in range(0, len(probmine[0])):
            for j in range(0, len(probmine[0])):
                if(probmine[0][i,j] > tmp_max and (i,j) not in self.positions_revealed):
                    tmp_max = probmine[0][i,j]
                    max_i = i
                    max_j = j
        return max_i, max_j
