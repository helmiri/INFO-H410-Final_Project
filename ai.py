import numpy as np
import random

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
        (x,y) = self.getMinProbPeri(peri, probmine, current_revealed)
        self.positions_revealed.append((x,y))
        return (x,y)

    def flag(self, probmine, peri, current_revealed):
        pos, pos_m = self.getMaxProbPeri(peri, probmine, current_revealed)
        if(pos[0] != None):
            self.positions_flaged.append((pos[0],pos[1]))
        return pos_m, pos

    """
    Return the position of the tile in the perimeter with the minimum probability of being a mine
    """
    def getMinProbPeri(self, peri, probmine, positions_revealed):
        prob_to_evaluate = {}
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed):
                prob_to_evaluate[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
        index = min(prob_to_evaluate, key=prob_to_evaluate.get)
        return index

    """
    Return the position of the tile in the perimeter with the maximum probability of being a mine
    """
    def getMaxProbPeri(self, peri, probmine, positions_revealed):
        prob_to_evaluate = {}; prob_to_mark = {}
        index = (None, None); index_m = (None, None)
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed and (pos[0], pos[1]) not in self.positions_flaged):
                prob_to_evaluate[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
            if((pos[0], pos[1]) not in positions_revealed):
                prob_to_mark[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
        prob_to_mark_fin = {k:v for k,v in prob_to_mark.items() if k not in prob_to_evaluate}
        if(len(prob_to_evaluate)!=0):
            index = max(prob_to_evaluate, key=prob_to_evaluate.get)
        if(len(prob_to_mark_fin)!=0):
            index_m = max(prob_to_mark_fin, key=prob_to_mark_fin.get)
        return index, index_m
