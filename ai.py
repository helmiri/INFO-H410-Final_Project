import numpy as np
import random

class AI:
    def __init__(self):
        self.boardsize = 0
        self.positions_revealed = []
        self.positions_flaged = []

    def setboardsize(self, size):
        """
        Set the boardsize attribut to the given size
        Parameters
        ----------
        size : int
            size of the board (Ex: 8, 16 or 24)
        """
        self.boardsize = size

    def reset(self):
        """
        Reset the list attributs of the class
        """
        self.positions_revealed = []
        self.positions_flaged = []

    def act(self, probmine, peri, current_revealed):
        """
        Return the position of the next tile where the AI will click

        Parameters
        ----------
        probmine : matrix
            matrix of the probability of getting a mine in each positions
        peri : list
            list of the (x,y) positions of the tiles in the perimeter of revealed tiles
        current_revealed : list
            list of the tiles already revealed
        """
        (x,y) = self.getMinProbPeri(peri, probmine, current_revealed)
        self.positions_revealed.append((x,y))
        return (x,y)

    def flag(self, probmine, peri, current_revealed):
        """
        Return the position of the next tile where the AI will put a flag and the
        position of the next tile where the AI will mark

        Parameters
        ----------
        probmine : matrix
            matrix of the probability of getting a mine in each positions
        peri : list
            list of the (x,y) positions of the tiles in the perimeter of revealed tiles
        current_revealed : list
            list of the tiles already revealed
        """
        pos, pos_m = self.getMaxProbPeri(peri, probmine, current_revealed)
        if(pos[0] != None):
            self.positions_flaged.append((pos[0],pos[1]))
        return pos_m, pos

    def getMinProbPeri(self, peri, probmine, positions_revealed):
        """
        Return the position of the tile in the perimeter with the minimum probability of being a mine

        Parameters
        ----------
        peri : list
            list of the (x,y) positions of the tiles in the perimeter of revealed tiles
        probmine : matrix
            matrix of the probability of getting a mine in each positions
        positions_revealed : list
            list of the tiles already revealed
        """
        prob_to_evaluate = {}
        for pos in peri:
            if((pos[0], pos[1]) not in positions_revealed):
                prob_to_evaluate[(pos[0], pos[1])] = probmine[0][pos[1], pos[0]]
        index = min(prob_to_evaluate, key=prob_to_evaluate.get)
        return index

    def getMaxProbPeri(self, peri, probmine, positions_revealed):
        """
        Return the position of the tile not flagged in the perimeter with the maximum probability of being a mine
        And return the position of the tile in the perimeter with the maximum probability of being a mine

        Parameters
        ----------
        peri : list
            list of the (x,y) positions of the tiles in the perimeter of revealed tiles
        probmine : matrix
            matrix of the probability of getting a mine in each positions
        positions_revealed : list
            list of the tiles already revealed
        """
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
