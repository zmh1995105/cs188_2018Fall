# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import operator
class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            states = self.mdp.getStates()
            valueOfState = util.Counter()
            for state in states:
                bestAction = self.computeActionFromValues(state)
                valueOfState[state] = self.computeQValueFromValues(state, bestAction)
            self.values = valueOfState.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          # Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = E_s'(T(s, a, s')*(R(s, a, s') + discount * V(s))
        Q = 0
        if action:
            tStateProb = self.mdp.getTransitionStatesAndProbs(state, action)

            for nextState, T in tStateProb:
                R = self.mdp.getReward(state, action, nextState)
                V = self.getValue(nextState)
                Q += T * (R + self.discount * V)
        # print("Q = {}".format(Q))
        return Q
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Vπ = argmax_a(E_a(Q(s, a))
        actions = self.mdp.getPossibleActions(state)
        if len(actions):
            counter = util.Counter()
            for action in actions:
                counter[action] = self.computeQValueFromValues(state, action)
            bestAction = counter.argMax()
            return bestAction
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # If the state picked for updating is terminal, nothing happens in that iteration.
        states = self.mdp.getStates()

        # iterate to update the state if is not terminal and
        # if it is the last state, then update the first state
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                bestAction = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state, bestAction)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getDiff(self, state):
        """
        calculate the diff of current state
        :param state: current state
        :return: diff
        """
        currentValue = self.getValue(state)
        bestAction = self.computeActionFromValues(state)
        highestValue = self.computeQValueFromValues(state, bestAction)
        diff = abs(currentValue - highestValue)
        return diff

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        queue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = {state: set() for state in states}
        for state in states:
            # 1. get the prodecessors of state
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                tStateProb = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in tStateProb:
                    predecessors[nextState].add(state)

            # 2. get diff of state and push in to queue
            if not self.mdp.isTerminal(state):
                diff = self.getDiff(state)
                queue.push(state, -diff)

        # 3. iterate to update the self.values
        for i in range(self.iterations):
            if queue.isEmpty():
                return

            currentState = queue.pop()
            # 3.1 if not terminal state, update the value
            if not self.mdp.isTerminal(currentState):
                bestAction = self.computeActionFromValues(currentState)
                self.values[currentState] = self.computeQValueFromValues(currentState, bestAction)

            # 3.2 get the predecessors of currentState and push the diff of p
            # if the diff > theta and diff > current diff of p in the queue
            for p in predecessors[currentState]:
                pDiff = self.getDiff(p)
                if pDiff > self.theta:
                    queue.update(p, -pDiff)




