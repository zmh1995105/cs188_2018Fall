# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"

        # print("newPos: ", newPos)
        # print("newFood: ", newFood)


        WEIGHT_GHOST, WEIGHT_FOOD = 10, 10

        distanceToGhost = []
        for index, newScaredTime in enumerate(newScaredTimes):
            # print("ScaredTimes: ", newScaredTimes)
            # print("GhostState {}: ".format(index), newGhostStates[index])
            if newScaredTime == 0:
                notScaredGhosePos = newGhostStates[index].getPosition()
                distance = manhattanDistance(newPos, notScaredGhosePos)
                distanceToGhost.append(distance)
            if newScaredTime > 0:
                distanceToGhost.append(1000)

        distanceToFood = [manhattanDistance(newPos, FoodPos) for FoodPos in newFood.asList()]

        score = successorGameState.getScore()
        if distanceToFood:
            score += WEIGHT_FOOD / min(distanceToFood)

        if min(distanceToGhost) > 0:
            score -= WEIGHT_GHOST / min(distanceToGhost)
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getSuccessors(self, gameState, agentIndex):
        return [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]

    def isTerminal(self, gameState, depth, agentIndex):
        return self.depth == depth or gameState.isWin() or gameState.isLose()



class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, depth, agentIndex):
            if self.isTerminal(gameState, depth, 0):
                return self.evaluationFunction(gameState)
            # pacman ==> max
            elif agentIndex == 0:
                v = float("-inf")
                for successor in self.getSuccessors(gameState, agentIndex):
                    v = max(v, minimax(successor, depth, 1))
                return v
            # ghostAgent ==> min
            else:
                v = float("inf")
                for successor in self.getSuccessors(gameState, agentIndex):
                    # the last agent ==> creat next depth
                    if agentIndex == gameState.getNumAgents()-1:
                        v = min(v, minimax(successor, depth+1, 0))
                    else:
                        # compare with next ghost
                        v = min(v, minimax(successor, depth, agentIndex+1))
                return v

        return max(gameState.getLegalActions(0), key = lambda action: minimax(gameState.generateSuccessor(0, action), 0, 1))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaPrune(gameState, depth, agentIndex, alpha, beta):
            if self.isTerminal(gameState, depth, 0):
                return self.evaluationFunction(gameState)
            # pacman ==> max
            elif agentIndex == 0:
                v = float("-inf")
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, alphaBetaPrune(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v

            # ghostAgent ==> min
            else:
                v = float("inf")
                for action in gameState.getLegalActions(agentIndex):
                    # the last agent ==> creat next depth
                    if agentIndex == gameState.getNumAgents()-1:
                        v = min(v, alphaBetaPrune(gameState.generateSuccessor(agentIndex, action), depth+1, 0, alpha, beta))
                    else:
                        # compare with next ghost
                        v = min(v, alphaBetaPrune(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(v, beta)
                return v

        alpha, beta = -float("inf"), float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphaBetaPrune(successor, 0, 1, alpha, beta)
            if value > alpha:
                bestAction = action
                alpha = value
        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        def expectimax(gameState, depth, agentIndex):
            if self.isTerminal(gameState, depth, 0):
                return self.evaluationFunction(gameState)
            # pacman ==> max
            elif agentIndex == 0:
                v = float("-inf")
                for successor in self.getSuccessors(gameState, agentIndex):
                    v = max(v, expectimax(successor, depth, 1))
                return v
            # ghostAgent ==> min
            else:
                v = 0
                for successor in self.getSuccessors(gameState, agentIndex):
                    # the last agent ==> creat next depth
                    if agentIndex == gameState.getNumAgents()-1:
                        v += expectimax(successor, depth+1, 0)
                    else:
                        # compare with next ghost
                        v += expectimax(successor, depth, agentIndex+1)
                return v

        return max(gameState.getLegalActions(0), key = lambda action: expectimax(gameState.generateSuccessor(0, action), 0, 1))







def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    WEIGHT_GHOST, WEIGHT_FOOD = 10, 10

    distanceToGhost = []
    for index, newScaredTime in enumerate(newScaredTimes):
        # print("ScaredTimes: ", newScaredTimes)
        # print("GhostState {}: ".format(index), newGhostStates[index])
        if newScaredTime == 0:
            notScaredGhosePos = newGhostStates[index].getPosition()
            distance = manhattanDistance(newPos, notScaredGhosePos)
            distanceToGhost.append(distance)
        if newScaredTime > 0:
            distanceToGhost.append(1000)

    distanceToFood = [manhattanDistance(newPos, FoodPos) for FoodPos in newFood.asList()]

    score = successorGameState.getScore()
    if distanceToFood:
        score += WEIGHT_FOOD / min(distanceToFood)

    if min(distanceToGhost) > 0:
        score -= WEIGHT_GHOST / min(distanceToGhost)
    return score
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
