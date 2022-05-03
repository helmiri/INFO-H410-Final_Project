import itertools
import random
from typing import Dict, Iterable

from tile import Tile

"""
    Return probability board
    - revealed_values: List of values
    - neighborhoods: List of lists of values where list i corresponds to the neighborhood of value i in revealed_values
    - perimeter: Perimeter of the revealed values where every key is the coordinates (x, y) of the tile
"""
def naive(revealed_values: Iterable[int], neighborhoods: Iterable[Iterable[Tile]],  perimeter: Dict[Tile, int]):
    for i, neighborhood in enumerate(neighborhoods):
        if revealed_values[i].is_start:
            continue
        for placement in generate_bomb_placements(len(neighborhood), revealed_values[i].get_value()):
            for j, tile in enumerate(placement):
                if tile == 1 and neighborhood[j] in perimeter:
                    perimeter[neighborhood[j]] += 1

    minval = min(perimeter.values())
    res = [k for k, v in perimeter.items() if v==minval]
    return res

def generate_bomb_placements(neighborhood_size: int, num_bombs: int):
    for bomb_popsitions in itertools.combinations(range(neighborhood_size), num_bombs):
        temp = [0] * neighborhood_size
        for i in bomb_popsitions:
            temp[i] = 1
        yield temp

"""
    If number of hidden tiles in the neighborhood equals the value of the tile, flag all tiles surrounding it as bombs.
"""
def rule_1(revealed_values: Iterable[int], neighborhoods: Iterable[Iterable[Tile]]):
    for i, neighborhood in enumerate(neighborhoods):
        length = len(neighborhood)
        for tile in neighborhood:
            if tile.is_revealed:
                length -= 1
        if revealed_values[i].get_value() == length:
            for tile in neighborhood:
                if not tile.is_revealed and not tile.is_flagged:
                    return tile
    return None

"""
    If the number of flagged tiles in the neighborhood equals the value of the tile, all hidden tiles in the neighborhood are safe
"""
def rule_2(revealed_values: Iterable[int], neighborhoods: Iterable[Iterable[Tile]]):
    for i, neighborhood in enumerate(neighborhoods):
        flags = 0
        for tile in neighborhood:
            if tile.is_flagged:
                flags += 1
        if revealed_values[i].get_value() == flags:
            for tile in neighborhood:
                if not tile.is_flagged and not tile.is_revealed:
                    return tile
    return None