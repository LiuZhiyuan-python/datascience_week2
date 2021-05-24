#!/usr/bin/env python
# _*_coding:utf-8_*_
# vim:fenc=utf-8

"""
"""
from typing import List, Tuple, Union

from cyaron import Vector


class Node:
    def __init__(self, value, current_dim):
        self.value = value
        self.current_dim = current_dim
        self.left = None
        self.right = None
        self.father: Node = None

    def __str__(self):
        return str(self.value)

    @property
    def brother(self):
        if self.father and self == self.father.left:
            return self.father.right
        elif self.father and self == self.father.right:
            return self.father.left
        else:
            return None


class KDTree:
    def __init__(self, root_node, total_dims, values, sorted_indexes):
        self.root_node = root_node
        self.total_dims = total_dims
        self.values = values
        self.sorted_indexes = sorted_indexes

    @classmethod
    def build(cls, values):
        total_dims = len(values[0])
        sorted_indexes = []
        for dim in range(total_dims):
            tmp_indexes = sorted(range(len(values)), key=lambda index: values[index][dim])
            sorted_indexes.append(tmp_indexes)

        def _build(indexes: set, dim):
            dim = dim % total_dims
            # real_indexes = sorted(indexes, key=lambda index: values[index][dim])
            real_indexes = [index for index in sorted_indexes[dim] if index in indexes]
            if len(real_indexes) == 0:
                return None
            elif len(real_indexes) == 1:
                return Node(value=values[real_indexes[0]], current_dim=dim)

            # if _middle 靠左侧, 那么生成出来的 kd 树总是尽量使得右子树是满的
            # 如果靠右侧, 生成出来的树尽量使得左侧是满的
            # 下述逻辑: 尽量往右找第一个不等于_middle位置的数字的位置, 可以为_middle 也可以是 最后一个位置
            # _middle = (len(real_indexes) - 1) // 2
            _middle = len(real_indexes) // 2
            for middle in range(_middle, len(real_indexes), 1):
                if middle == len(real_indexes) - 1 \
                        or values[real_indexes[middle + 1]][dim] != values[real_indexes[_middle]][dim]:
                    break
            root = Node(values[real_indexes[middle]], current_dim=dim)
            left_node = _build(set(real_indexes[:middle]), dim + 1)
            right_node = _build(set(real_indexes[middle + 1:]), dim + 1)

            root.left = left_node
            if left_node:
                left_node.father = root
            root.right = right_node
            if right_node:
                right_node.father = root
            return root

        root = _build(set(range(len(values))), dim=0)
        return cls(root, values=values, sorted_indexes=sorted_indexes, total_dims=total_dims)

    def square_distance_fn(self, xs: tuple, ys: tuple):
        distance = 0
        for x, y in zip(xs, ys):
            distance += (x - y) ** 2
        return distance

    def print_tree(self):
        nodes: List[Node] = [self.root_node]
        while len(nodes) != 0:
            node = nodes.pop(0)
            print(f'  {node} ')
            print(f' /{" " * len(node.value) * 2}\\')
            print(f'{node.left} {node.right}')
            print(f'-' * 30)
            if node.left:
                nodes.append(node.left)
            if node.right:
                nodes.append(node.right)

    def find_nearest_point_force(self, to_be_found):

        def _bfs(nodes: List[Node], value_vs_dist: List[Tuple[Tuple, int]]):
            while len(nodes) != 0:
                node = nodes.pop(0)
                value_vs_dist.append((node.value, self.square_distance_fn(to_be_found, node.value)))
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)

        value_vs_dist = []
        _bfs([self.root_node], value_vs_dist)
        min_value, min_distance = sorted(value_vs_dist, key=lambda vd: vd[1])[0]

        return min_value, min_distance

    def find_nearest_point(self, to_be_found: Union[Tuple, List]):
        def _find_leaf_node(root: Node) -> Node:
            dim = root.current_dim

            if root.left:
                if to_be_found[dim] <= root.value[dim] or root.right is None:
                    return _find_leaf_node(root.left)
                elif root.right:
                    return _find_leaf_node(root.right)
                else:
                    return root

            if root.right:
                if to_be_found[dim] > root.value[dim] or root.left is None:
                    return _find_leaf_node(root.right)
                elif root.left:
                    return _find_leaf_node(root.left)
                else:
                    return root

            return root

        best_node = _find_leaf_node(self.root_node)
        min_dist = float("inf")
        que = [(self.root_node, best_node)]

        while que:
            root_node, current_node = que.pop(0)
            root_distance = self.square_distance_fn(root_node.value, to_be_found)
            if root_distance < min_dist:
                min_dist = root_distance
                best_node = root_node

            while current_node != root_node:
                dist = self.square_distance_fn(current_node.value, to_be_found)
                if dist < min_dist:
                    min_dist = dist
                    best_node = current_node

                if current_node.brother and min_dist \
                        > (to_be_found[current_node.father.current_dim] - current_node.father.value[
                    current_node.father.current_dim]) ** 2:
                    que.append((current_node.brother, _find_leaf_node(current_node.brother)))
                current_node = current_node.father

        return best_node, min_dist


if __name__ == '__main__':
    values = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    values = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
    values = Vector.random(10, [10, 10, 15])
    to_be_found = Vector.random(1, [90, 100, 30])[0]

    # values = [[7, 0, 2], [1, 8, 1], [5, 4, 2], [10, 5, 6], [5, 1, 10], [8, 0, 4], [6, 4, 12], [8, 3, 15], [4, 8,
    # 0], [9, 5, 5]]
    # to_be_found = [71, 39, 30]
    # to_be_found = (3, 4, 4)
    kdtree = KDTree.build(values)
    kdtree.print_tree()

    founded_point, min_dist = kdtree.find_nearest_point_force(to_be_found)
    founded_point2, min_dist2 = kdtree.find_nearest_point(to_be_found)
    founded_point2 = founded_point2.value

    assert list(founded_point) == list(founded_point2)
    pass
    ...