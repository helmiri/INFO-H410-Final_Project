from __future__ import annotations
from queue import Queue
from PyQt5.QtWidgets import QApplication
from typing import Dict, Iterable, List, Set, Tuple

from tile import Tile


class TileTree:
    """
    Binary Tree node
    This class will be used in the construction of a binary tree where each branch corresponds
    to a possible mine placement.
    """

    def __init__(self, value=None, tile=None, parent=None, neighbors=[]) -> None:
        """
            - value    : int indicating whether this tile should contain a mine or not. 0 = No mine or 1 = Mine 
            - tile     : The hidden Tile being considered
            - parent   : Parent TileTree node
            - neighbors: Revealed neighboring Tiles
        """
        self.value = value
        self.neighbors = neighbors
        self.tile = tile
        self.parent = parent

        # Children
        self.left = None   # Does not contain a mine
        self.right = None  # Contains a mine

    def get_tile(self) -> Tile:
        return self.tile

    def get_value(self) -> int:
        return self.value

    def get_left(self) -> TileTree:
        return self.left

    def get_right(self) -> TileTree:
        return self.right

    def get_neighbors(self) -> List[Tile]:
        return self.neighbors

    def add_child(self, child) -> None:
        if child.get_value() == 0:
            self.left = child
        else:
            self.right = child

    def remove_child(self, child) -> None:
        if child.get_value() == 0:
            self.left = None
        else:
            self.right = None

        if self.value is not None:
            self.prune()

    def has_children(self) -> bool:
        if self.left is not None or self.right is not None:
            return True
        return False

    def prune(self) -> None:
        """
        Remove unneeded branches. When a mine placement is invalid, it will be removed from the tree.
        Then, if the parent node has no children, the parent will remove itself and so on.
        This results in invalid branches being removed
        """
        if self.left is None and self.right is None:
            self.parent.remove_child(self)

    def __hash__(self) -> int:
        return hash((self.tile, self.value, self.parent))

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, Tile):
            return self.tile == other.tile
        return self.tile == other.tile and self.value == other.value

    def __ne__(self, other):
        return not(self == other)


def rule_1(revealed_values: Iterable[Tile], neighborhoods: Iterable[Iterable[Tile]]) -> Tile:
    """
        If number of hidden tiles in the neighborhood equals the value of the tile, flag all tiles surrounding it as bombs.
            args:
                - revealed_values: List of revealed tiles
                - neighborhoods  : List of lists of revealed tiles where the i-th list is the neighborhood for the i-th tile
                                in revealed_values
            returns:
                - The tile if applicable or None if the rule could not be applied
    """
    for i, neighborhood in enumerate(neighborhoods):
        length = len(neighborhood)
        flags = 0
        for tile in neighborhood:
            if tile.is_revealed:
                length -= 1
            elif tile.is_flagged:
                flags += 1
        if revealed_values[i].get_value() - flags == length - flags:
            for tile in neighborhood:
                if not tile.is_revealed and not tile.is_flagged:
                    return tile
    return None


def rule_2(revealed_values: Iterable[Tile], neighborhoods: Iterable[Iterable[Tile]]) -> Tile:
    """
        If the number of flagged tiles in the neighborhood equals the value of the tile, all hidden tiles in the neighborhood are safe
            args:
                    - revealed_values: List of revealed tiles
                    - neighborhoods  : List of lists of revealed tiles where the i-th list is the neighborhood for the i-th tile
                                    in revealed_values
            returns:
                - The tile if applicable or None if the rule could not be applied
    """
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


def rule_3(perimeter: Set[Tile], revealed: Set[Tile]) -> Tuple[List[Tile], List[Tile], int]:
    """
        Exhaustive search. Uses a binary tree to build all valid mine placements in perimeter
        args:
            - perimeter: Set of tiles in the revealed perimeter
            - revealed : Set of revealed tiles adjacent to perimeter tiles
        returns:
            - free : List of tiles safe to click
            - mines: List of tiles safe to flag
            - certainty: 1 = We are certain that all tiles returned are safe to click, 0 = We are not
    """
    hidden_processed = set()
    root = TileTree()
    parent_queue = {root}
    current_queue = set()
    perimeter_queue = set()
    certainty = 0
    while len(perimeter) > 0:

        if len(perimeter_queue) > 0:
            tile = perimeter_queue.pop()
        else:
            tile = perimeter.pop()
        if tile in hidden_processed:
            continue

        hidden_processed.add(tile)

        # Separate hidden tiles from revealed tiles in the neighborhood
        # The algorithm doesn't care about the order of tiles visited but it is more efficient
        # if it visits them in order.

        neighbors = list()
        for neighbor in tile.neighbors:
            if neighbor in revealed and not neighbor.is_start:
                neighbors.append(neighbor)
            elif not neighbor.is_revealed and neighbor in perimeter and neighbor not in hidden_processed:
                perimeter_queue.add(neighbor)

        # Update the tree
        # Each tile is inserted into tree with value 0 = no mine, value 1 = mine to create possible arrangements
        # parent_queue contains the terminal nodes of the tree. These are the new parents.
        current_queue.clear()
        while len(parent_queue) > 0:
            parent = parent_queue.pop()
            for value in [0, 1]:
                # If the tile is flagges, we only insert it once. This prevents unneeded checks.
                if value == 0 and tile.is_flagged:
                    continue
                new_node = TileTree(value, tile, parent, neighbors)
                current_queue.add(new_node)
                parent.add_child(new_node)

        removed = set()  # Will store the terminal nodes removed
        is_valid(root, removed)

        # Calculate the probabilities of each tile containing a mine.
        # This check is done at each iteration because it allows us to stop early when we detect that
        # the probability of a tile containing a mine is 0.
        # This happens when all possibilities of where it contains a mine have been discarded.
        nodes = merge_branches(root)
        if len(nodes) != 0:
            minval = min(nodes.values())
            free = list()
            if minval == 0:
                tiles = [k for k, v in nodes.items() if v == minval]
                free = [
                    candidate for candidate in tiles if candidate not in current_queue]
                certainty = 1
                mines = [k for k, v in nodes.items() if v == 1]
                return free, mines, certainty

        # Discard removed nodes from the queue.
        # New terminal nodes become parents for the next iteration
        parent_queue = current_queue - removed

    nodes = merge_branches(root)
    minval = min(nodes.values())
    if minval == 0:
        certainty = 1
    free = [k for k, v in nodes.items() if v == minval]
    mines = [k for k, v in nodes.items() if v == 1]
    return free, mines, certainty


def merge_branches(tree: TileTree) -> Dict:
    """
    For each tile, count occurrences where it is a mine
        args:
            - tree: Root node of the tree
        returns:
            - Dictionary containing as key the tile and as value its probability of containing a mine
    """
    nodes = dict()
    occurrences = dict()
    queue = Queue()
    queue.put(tree)
    while not queue.empty():
        node = queue.get()
        for child in [node.get_left(), node.get_right()]:
            if child is not None:
                if child.has_children():
                    queue.put(child)
                    tile = child.get_tile()
                    if tile not in nodes:
                        nodes[tile] = 0
                        occurrences[tile] = 0
                    nodes[tile] += child.get_value()
                    occurrences[tile] += 1
    for key in nodes:
        nodes[key] = nodes[key] / occurrences[key]
    return nodes


def is_valid(tree: TileTree, removed: Set[Tile]) -> None:
    """
    Check the validity of the tree and prune invalid branches.
        args:
            - tree: Root node from which to explore
            - removed: Set of tiles where removed nodes will be stored
    """

    # We have reached the end
    if tree is None:
        return

    # Mark tiles we traverse
    if tree.get_value() is not None:
        tree.get_tile().mark(tree.get_value())

    # This is a terminal node. Check its validity.
    if tree.get_left() is None and tree.get_right() is None:
        QApplication.processEvents()
        if not is_satisfied(tree):
            # Remove node from tree
            tree.remove_child(tree)
            removed.add(tree)
    else:
        is_valid(tree.get_left(), removed)
        is_valid(tree.get_right(), removed)

    # Unmark when backtracking
    if tree.get_value() is not None:
        tree.get_tile().unmark()


def is_satisfied(tree: TileTree) -> bool:
    """
    Checks that this does not rbeak the rules of the game
        args: 
            - tree: Terminal node of the tree
        returns:
            - True if the placement is valid, False otherwise
    """
    # For each revealed neighbor
    for neighbor in tree.get_neighbors():
        mines = 0
        free = 0
        # Get the hidden tiles in its neighborhood
        hidden_neighborhood = [
            n for n in neighbor.neighbors if not n.is_revealed]

        # For each hidden tile in the neighborhood
        for hidden_neighboring in hidden_neighborhood:
            # If it is marked, determine whether it is a mine or not
            if hidden_neighboring.is_marked():
                if hidden_neighboring.get_mark() == 0:
                    free += 1
                else:
                    mines += 1

        # Calculate how many mines are left to be placed according to the current state
        remaining = neighbor.get_value() - mines
        if remaining < 0:
            # We placed too many mines
            return False
        if remaining > len(hidden_neighborhood) - mines - free:
            # We didn't place enough mines because there aren't enough unmarked tiles in its neighborhood
            return False
    return True
