# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


class Node(object):
    def __init__(self, location, direction=list(), priority=0, oldHeur=None, oldCost=0):
        self.location = location
        self.direction = direction
        self.priority = priority
        self.oldHeur = oldHeur
        self.oldCost = oldCost

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return searchFunction(problem, lambda x, y: x - y)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return searchFunction(problem, lambda x, y: x + y)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return searchFunction(problem, lambda x, y: x + y)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    return searchFunction(problem, lambda x, y: x + y, heuristic)

# def searchFunction(problem, func, heuristic=nullHeuristic):
#     queue = util.PriorityQueue()
#     visited = set()
#     start_node = Node(problem.getStartState())
#     queue.push(start_node, start_node.priority)
#
#
#     while not queue.isEmpty():
#         currentNode = queue.pop() # type: Node
#         # check if the heuristic is consistent
#         currentHeur = heuristic(currentNode.location, problem)
#         if currentNode.oldHeur != None:
#             if currentHeur + currentNode.oldCost < currentNode.oldHeur:
#                 raise Exception("Is not consistent!!")
#
#         currentNode.oldHeur = currentHeur
#         print("The heuristic is consistent ==> newHeur:{}, oldHeur:{}, oldCost:{}".format(currentHeur, currentNode.oldHeur, currentNode.oldCost))
#
#         if problem.isGoalState(currentNode.location):
#             return currentNode.direction
#         if currentNode.location not in visited:
#             visited.add(currentNode.location)
#             for successor in problem.getSuccessors(currentNode.location):
#                 cost = successor[2]
#                 newPriority = func(currentNode.priority, cost)
#                 new_node = Node(successor[0], currentNode.direction + [successor[1]], newPriority, currentHeur, cost)
#                 queue.push(new_node, new_node.priority + heuristic(successor[0], problem))
#     return []

def searchFunction(problem, func, heuristic=nullHeuristic):
    queue = util.PriorityQueue()
    visited = list()
    start_node = Node(problem.getStartState())
    queue.push(start_node, start_node.priority)


    while not queue.isEmpty():
        currentNode = queue.pop() # type: Node

        if problem.isGoalState(currentNode.location):
            return currentNode.direction
        if currentNode.location not in visited:
            visited.append(currentNode.location)
            for successor in problem.getSuccessors(currentNode.location):
                cost = successor[2]
                newPriority = func(currentNode.priority, cost)
                new_node = Node(successor[0], currentNode.direction + [successor[1]], newPriority)
                queue.push(new_node, new_node.priority + heuristic(successor[0], problem))
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
