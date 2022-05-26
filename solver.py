import itertools
from queue import Queue
from PyQt5.QtCore import QEventLoop, QTimer
from typing import Dict, Iterable

from tile import Tile

"""
    DEPRECATED: see rule3

    Return probability board
    - revealed_values: List of values
    - neighborhoods: List of lists of values where list i corresponds to the neighborhood of value i in revealed_values
    - perimeter: Perimeter of the revealed values where every key is the coordinates (x, y) of the tile
"""


def naive(revealed_values: Iterable[int], neighborhoods: Iterable[Iterable[Tile]],  perimeter: Dict[Tile, int]):
    for i, neighborhood in enumerate(neighborhoods):
        if revealed_values[i].is_start:
            continue
        placements = get_placements(
            len(neighborhood), revealed_values[i].get_value(), neighborhood)
        for placement in placements:
            for j, tile in enumerate(placement):
                if tile == 1 and neighborhood[j] in perimeter:
                    perimeter[neighborhood[j]] += 1

    minval = min(perimeter.values())
    res = [k for k, v in perimeter.items() if v == minval]
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
    Check validity of placements
"""


def placement_check(placement, neighborhood):
    for i, neighbor in enumerate(neighborhood):
        if placement[i] == 1 and neighbor.is_revealed or neighbor.is_flagged and placement[i] == 0:
            return False
    return True


"""
    Generate and validate mine placements
"""


def get_placements(size, value, neighborhood):
    placements = list()
    # Generate mine placements for current tile
    for placement in generate_bomb_placements(size, value):
        if placement_check(placement, neighborhood):
            placements.append(placement)
    return placements


# DEPRECATED
# def rule_3(revealed, neighborhoods):
#     tiles_neighbors = dict(zip(revealed, neighborhoods))

#     for tile in tiles_neighbors:
#         neighborhood = tiles_neighbors[tile]
#         temp_vec = [t.is_revealed for t in neighborhood]
#         if temp_vec.count(True) == len(temp_vec):
#             continue

#         tile_placements = get_placements(
#             len(temp_vec), tile.value, neighborhood)

#         neighbor_placements = list(list() * len(neighborhood))
#         for i, neighbor_tile in enumerate(neighborhood):
#             neighbor_neighborhood = tiles_neighbors[neighbor_tile]
#             targets = list(shared_tile for shared_tile in neighbor_neighborhood if not shared_tile.is_revealed and shared_tile in neighborhood
#                            )

#             temp_vec = [t.is_revealed for t in neighborhood]
#             if not neighbor_tile.is_revealed or temp_vec.count(True) == len(temp_vec):
#                 continue

#             neighbor_placements[i] = get_placements(
#                 len(temp_vec), tile.value, neighborhood)


"""
    Revealed tiles are wrapped with this class. Allows tracking of how many remaining mines must be placed in its neighborhood
"""


class LinkedTile:
    def __init__(self, tile) -> None:
        self.tile = tile
        self.remaining = get_remaining(tile)
        self.count = 0
        self.children = list()

    def get_remain(self):
        return self.remaining - self.count

    """
        Increase mine count
    """

    def increase_count(self):
        if self.get_remaining_bombs() == 0:
            return False
        self.count += 1
        return True

    def decrease_count(self):
        self.count = max(self.count - 1, 0)

    def get_count(self):
        return self.count

    def reset_count(self):
        self.count = 0

    """
        Remaining mines to be placed in the neighborhood of this tile
    """

    def get_remaining_bombs(self):
        flags = 0
        for neighbor in self.tile.neighbors:
            if neighbor.is_flagged:
                flags += 1
        return self.tile.get_value() - self.count

    def get_hidden_neighbors(self):
        hidden = set()
        for tile in self.tile.neighbors:
            if not tile.is_revealed:
                hidden.add(tile)
        return hidden

    def __hash__(self):
        return hash(self.tile)

    def __eq__(self, other):
        if other is None:
            return False
        return self.tile == other.tile

    def __ne__(self, other):
        return not(self == other)


"""
    Binary Tree node
"""


class TileTree:
    def __init__(self, value=None, tile=None, parent=None, neighbors=None) -> None:
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

    def remove_child(self, child):
        if child.get_value() == 0:
            self.left = None
        else:
            self.right = None

        if self.value is not None:
            self.prune()

    def remove_self(self):
        self.parent.remove_child(self)

    """
        Remove unneeded branches. This happens when a mine placement is invalid. The corresponding branch will be removed
    """

    def prune(self):
        if self.left is None and self.right is None:
            self.parent.remove_child(self)

    def update_branch_count(self, count):
        self.branch_count += count

    def __hash__(self):
        return hash((self.tile, self.value))

    def __eq__(self, other):
        if other is None:
            return False
        return self.tile == other.tile and self.value == other.value

    def __ne__(self, other):
        return not(self == other)


def rule3(perimeter, revealed):
    linked_tiles = dict()
    hidden_processed = set()
    root = TileTree()
    parent_queue = {root}
    current_queue = set()

    perimeter_queue = set()
    while len(perimeter) > 0:
        if len(perimeter_queue) > 0:
            tile = perimeter_queue.pop()
        else:
            tile = perimeter.pop()
        if tile in hidden_processed:
            continue

        hidden_processed.add(tile)

        # Preprocess Neighbors -> Convert revealed tiles into LinkedTile
        neighbors = list()
        for neighbor in tile.neighbors:
            if neighbor in revealed:
                if neighbor in linked_tiles:
                    tmp = linked_tiles[neighbor]
                else:
                    tmp = LinkedTile(neighbor)
                    linked_tiles[neighbor] = tmp
                neighbors.append(tmp)
            elif not neighbor.is_revealed and neighbor in perimeter and neighbor not in hidden_processed:
                perimeter_queue.add(neighbor)

        # Create and add nodes to tree -> Convert hidden tiles in perimeter into tree nodes
        current_queue.clear()
        while len(parent_queue) > 0:
            parent = parent_queue.pop()
            branch_count = 2
            for value in [0, 1]:  # Each tile is inserted into tree with value 0 = no mine, value 1 = mine to create possible arrangements
                if value == 0 and tile.is_flagged:
                    branch_count = 0
                    continue
                new_node = TileTree(value, tile, parent, neighbors)
                current_queue.add(new_node)
                parent.add_child(new_node)
                new_node.get_tile().unmark()
            root.update_branch_count(branch_count)

        removed = set()  # Track removed nodes
        is_valid(root, set(), removed)
        parent_queue = current_queue - removed  # Valid terminal nodes after removal

    nodes = merge_branches(root)
    minval = min(nodes.values())
    values = [k.get_tile() for k, v in nodes.items() if v == minval]
    return values


"""
    For each tile, count occurrences where it is a mine
"""


def merge_branches(tree):
    nodes = dict()
    queue = Queue()
    queue.put(tree)

    while not queue.empty():
        node = queue.get()

        for child in [node.get_left(), node.get_right()]:
            if child is not None:
                queue.put(child)
                if child not in nodes:
                    nodes[child] = 0
                nodes[child] += child.get_value()
    assert len(nodes) > 0
    return nodes

    # def rule3(revealed, values):
    #     minimum_frequency_tiles = list()
    #     for tile in values:
    #         intersect = set()
    #         for neighbor in get_revealed_neighbors(tile):
    #             if neighbor in values and neighbor != tile:
    #                 intersect.update(revealed[neighbor])

    #         intersect = set.intersection(set(revealed[tile]), intersect)

    #         # SOMETHING WRONG WITH INTERSECT
    #         #  WE HAVE COMMON PERIM TILES.
    #         if intersect == set():
    #             continue
    #         neighboring_hidden = list()
    #         linked_tiles = list()
    #         intersect = list(intersect)  # For iteration purposes
    #         for i, hidden in enumerate(intersect):
    #             neighboring_hidden.append(list())
    #             for neighbor in hidden.neighbors:
    #                 new_linked = LinkedTile(neighbor)
    #                 try:
    #                     index = linked_tiles.index(new_linked)
    #                     new_linked = linked_tiles[index]
    #                 except ValueError:
    #                     linked_tiles.append(new_linked)
    #                 neighboring_hidden[i].append(new_linked)

    #         frequency = [0 for i in intersect]
    #         for bomb_placement in generate_bomb_placements(len(intersect), get_remaining(tile)):
    #             if intersect[i].is_flagged:
    #                 continue
    #             valid_placement = True
    #             for i, space in enumerate(bomb_placement):
    #                 if space == 1:
    #                     # Cannot place mine here
    #                     if not notify_neighbors(neighboring_hidden[i], 1):
    #                         # Remove it from neighbors
    #                         notify_neighbors(neighboring_hidden[i], -1)
    #                         valid_placement = False
    #                         break
    #             if valid_placement:
    #                 frequency = [a + b for a, b in zip(frequency, bomb_placement)]

    #         if 0 in frequency:
    #             return intersect[frequency.index(0)]
    #         minimum_index = frequency.index(min(frequency))
    #         minimum_frequency_tiles.append(
    #             (frequency[minimum_index], intersect[minimum_index]))

    #     if len(minimum_frequency_tiles) == 0:
    #         pass
    #     current = minimum_frequency_tiles[0]
    #     for item in minimum_frequency_tiles:
    #         if item[0] < current[0]:
    #             current = item
    #         ###########################################################""
    #         # neighbors = set()
    #         # for tile in intersect:
    #         #     neighbors.update(tile.neighbors)
    #         # neighbors = list(set.intersection(
    #         #     neighbors, lookup_table.values()))  # Get revealed values

    #         # # WE HAVE REVEALED TILES SHARING PERIMETER
    #         # size = len(intersect)
    #         # current = 0
    #         # for revealed_tile in neighbors:
    #         #     # If spaces in intersect = hidden spaces where bombs could be placed, all are bombs
    #         #     remaining = get_remaining(revealed_tile)
    #         #     if remaining == size:
    #         #         while len(intersect) > 0:
    #         #             tile = intersect.pop()
    #         #             if tile in lookup_table:
    #         #                 return tile
    #         #     else:
    #         #         current = max(current, remaining)

    #     return random.choice(minimum_frequency_tiles)[1]


"""
    For each neighbor in the neighborhood, inform it that a mine has been placed
    - neighbors : LinkedTile list
"""


def notify_neighbors(neighbors, operation):
    for neighbor in neighbors:
        if operation == -1:
            neighbor.decrease_count()
        elif operation == 1 and not neighbor.increase_count():
            return False
    return True


def get_revealed_neighbors(tile):
    neighbors = list()
    for t in tile.neighbors:
        if t.is_revealed:
            neighbors.append(t)
    return neighbors


"""
    Calculate remaining mines to be placed around tile
"""


def get_remaining(tile):
    neighbors = tile.neighbors
    flags = 0
    revealed = 0
    for neighbor in neighbors:
        if neighbor.is_revealed:
            revealed += 1
        elif neighbor.is_flagged:
            flags += 1
    mines_to_place = tile.get_value() - flags
    return len(tile.neighbors) - revealed - mines_to_place


def is_valid(tree, traversed, removed):
    # Visualization timer
    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec_()
    if tree.get_left() is not None:
        # When we go left, we keep track of nodes traversed.
        # if neighborhood - nodes traversed < mines to be placed, invalid placement -> Remove child from tree
        left_child = tree.get_left()
        left_child.get_tile().mark(left_child.get_value())
        traversed.add(left_child)
        check = True
        for neighbor in left_child.get_neighbors():
            tmp = neighbor.get_hidden_neighbors() - traversed
            if len(tmp) == 0:
                if neighbor.get_remaining_bombs() > 0:
                    check = False
                    break
        if check:
            is_valid(left_child, traversed, removed)
        else:
            tree.remove_child(left_child)
            removed.add(left_child)
        traversed.remove(left_child)
        left_child.get_tile().unmark()
    if tree.get_right() is not None:
        # When we go right, we place a mine. Notify neighbors that a mine has been placed on the tile
        # If it is invalid, remove node.
        right_child = tree.get_right()
        right_child.get_tile().mark(right_child.get_value())
        if notify_neighbors(right_child.get_neighbors(), 1):
            is_valid(right_child, traversed, removed)
            notify_neighbors(right_child.get_neighbors(), -1)
        else:
            tree.remove_child(right_child)
            removed.add(right_child)
        right_child.get_tile().unmark()
