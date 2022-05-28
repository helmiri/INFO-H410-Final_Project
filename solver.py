from queue import Queue
from PyQt5.QtCore import QEventLoop, QTimer
from typing import Iterable


from tile import Tile



"""
    If number of hidden tiles in the neighborhood equals the value of the tile, flag all tiles surrounding it as bombs.
"""

def rule_1(revealed_values: Iterable[int], neighborhoods: Iterable[Iterable[Tile]]):
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



"""
    Revealed tiles are wrapped with this class. Allows tracking of how many remaining mines must be placed in its neighborhood
"""

"""
    Binary Tree node
"""
class TileTree:
    def __init__(self, value=None, tile=None, parent=None, neighbors=[]) -> None:
        self.value = value
        self.neighbors = neighbors
        self.tile = tile
        self.parent = parent

        self.left = None
        self.right = None

        self.branch_count = 0

    def get_tile(self):
        return self.tile

    def get_value(self):
        return self.value

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_neighbors(self):
        return self.neighbors

    def get_branch_count(self):
        return self.branch_count

    def add_child(self, child):
        if child.get_value() == 0:
            self.left = child
        else:
            self.right = child
        self.update_branch_count(1)

    def remove_child(self, child):
        if child.get_value() == 0:
            self.left = None
        else:
            self.right = None

        if self.value is not None:
            self.prune()
        self.update_branch_count(-child.get_branch_count())

    def has_children(self):
        if self.left is not None or self.right is not None:
            return True
        return False

    """
        Remove unneeded branches. This happens when a mine placement is invalid. The corresponding branch will be removed
    """

    def prune(self):
        if self.left is None and self.right is None:
            self.parent.remove_child(self)

    def update_branch_count(self, count):
        if self.parent is not None:
            self.branch_count += count
            self.parent.update_branch_count(count)
        else:
            self.branch_count += count

    def get_root_branches(self):
        if self.value is not None:
            return self.parent.get_root_branches()
        return self.branch_count

    def __hash__(self):
        return hash((self.tile, self.value, self.parent))

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(other, Tile):
            return self.tile == other.tile
        return self.tile == other.tile and self.value == other.value

    def __ne__(self, other):
        return not(self == other)


def rule3(perimeter, revealed):
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

        neighbors = list()
        for neighbor in tile.neighbors:
            if neighbor in revealed and not neighbor.is_start:
                neighbors.append(neighbor)
            elif not neighbor.is_revealed and neighbor in perimeter and neighbor not in hidden_processed:
                perimeter_queue.add(neighbor)

        current_queue.clear()
        while len(parent_queue) > 0:
            parent = parent_queue.pop()
            for value in [0, 1]:  # Each tile is inserted into tree with value 0 = no mine, value 1 = mine to create possible arrangements
                if value == 0 and tile.is_flagged:
                    continue
                new_node = TileTree(value, tile, parent, neighbors)
                current_queue.add(new_node)
                parent.add_child(new_node)

        removed = set()
        is_valid(root, removed)
        nodes = merge_branches(root)
        if len(nodes) != 0:
            minval = min(nodes.values())
            free = list()
            if minval == 0:
                tiles = [k for k, v in nodes.items() if v == minval]
                free = [candidate for candidate in tiles if candidate not in current_queue]
                certainty = 1
                mines = [k for k, v in nodes.items() if v == 1]
                return free, mines, certainty
        parent_queue = current_queue - removed

    nodes = merge_branches(root)
    minval = min(nodes.values())
    if minval == 0:
        certainty = 1
    free = [k for k, v in nodes.items() if v == minval]
    mines = [k for k, v in nodes.items() if v == 1]
    return free, mines, certainty


"""
    For each tile, count occurrences where it is a mine
"""


def merge_branches(tree):
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


def is_valid(tree, removed):
    if tree is None:
        return
    # Visualization timer
    
    # print(tree.get_root_branches())
    if tree.get_value() is not None:
        tree.get_tile().mark(tree.get_value())

    if tree.get_left() is None and tree.get_right() is None:
        #loop = QEventLoop()
        #QTimer.singleShot(10, loop.quit)
        #loop.exec_()
        if not is_satisfied(tree):
            tree.remove_child(tree)
            removed.add(tree)
    else:
        is_valid(tree.get_left(), removed)
        is_valid(tree.get_right(), removed)

    if tree.get_value() is not None:
        tree.get_tile().unmark()



def is_satisfied(tree):
    for neighbor in tree.get_neighbors():  # Revealed neighbors
        mines = 0
        free = 0
        hidden_neighborhood = [
            n for n in neighbor.neighbors if not n.is_revealed]
        for hidden_neighboring in hidden_neighborhood:
            if hidden_neighboring.is_marked():
                if hidden_neighboring.get_mark() == 0:
                    free += 1
                else:
                    mines += 1
        remaining = neighbor.get_value() - mines
        if remaining < 0:
            return False
        if remaining > len(hidden_neighborhood) - mines - free:
            return False
    return True
