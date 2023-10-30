#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import heapq
from functools import total_ordering
from math import inf as infinity
from typing import List

robot_x, robot_y = 0, 0
battery_x, battery_y = 0, 0
n, m = 0, 0

#up,left,right,down
order_of_traversal = [(-1, 0), (0, -1), (0, 1), (1, 0)]


#total ordering is used to make it comparable with only one comparing function
class Node:
    parent = None
    visited = False
    dist = infinity
    x, y = -1, -1

    def __init__(self, x, y, kind='empty'):
        self.x = x
        self.y = y
        self.kind = kind
        pass

    def __str__(self):
        return str(self.x) + " " + str(self.y)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Node):
            return o.x == self.x and o.y == self.y and o.dist == self.dist
        return False

    def __gt__(self, other):
        if self.dist + manhatan_dist(self.x, self.y, battery_x, battery_y) != other.dist + manhatan_dist(self.x, self.y,
                                                                                                   battery_x, battery_y):
            return self.dist + manhatan_dist(self.x, self.y, battery_x, battery_y) > other.dist + manhatan_dist(self.x,
                                                                                                          self.y,
                                                                                                          battery_x,
                                                                                                          battery_y)
        #the next lines are for direction condition
        if self.x != other.x:
            return self.x > other.x
        else:
            return self.y > other.y


board = [[Node(-1, -1)]]


def manhatan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


#check if we can move to an index
def is_movable(i, j):
    if i < 0 or j < 0 or i >= n or j >= m:
        return False
    if board[i][j].kind == 'obstacle' or board[i][j].visited == True:
        return False
    return True


#find movable adjacent cells and returns and array
def find_valid_adjacent(i, j):
    answer = []
    for traverse in order_of_traversal:
        ii, jj = i + traverse[0], j + traverse[1]
        if is_movable(ii, jj):
            if (board[i][j].dist + (1 if board[ii][jj].kind != 'Battery' else 0)) <= board[ii][jj].dist:
                answer.append(board[ii][jj])
    return answer


#initialize map and get input
def initialize():
    global robot_x, robot_y, battery_x, battery_y, n, m, board, visited, parent

    #getting input
    robot_x, robot_y = map(int, input().split())
    battery_x, battery_y = map(int, input().split())
    n, m = map(int, input().split())
    board = [[Node(j, i) for i in range(m)] for j in range(n)]

    #initialization
    for i in range(n):
        row = input()
        for j in range(m):
            board[i][j].kind = row[j]


def astar():
    expansion_path = []
    heap: List[Node] = []
    heapq.heappush(heap, board[robot_x][robot_y])
    board[robot_x][robot_y].dist = 0
    while len(heap) != 0:
        v = heapq.heappop(heap)
        v.visited = True
        expansion_path.append(str(v))
        if v.x == battery_x and v.y == battery_y:
            break
        for u in find_valid_adjacent(v.x, v.y):
            u.visited = True
            u.dist = v.dist + (1 if u.kind != "Battery" else 0)
            heapq.heappush(heap, u)
            u.parent = v

def print_path(target_x, target_y):  
    node = board[target_x][target_y]
    path = []
    while node is not None:
        path.append(str(node))
        node = node.parent
    for robot in reversed(path):
        print(robot)
        
        
if __name__ == "__main__":
    initialize()
    astar()
    print_path(battery_x, battery_y)


# In[ ]:




