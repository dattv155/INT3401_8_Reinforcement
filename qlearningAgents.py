# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Create dictionary to store values of state-action pairs
        self.count = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Returns Q(state,action).
        # 0 if we have never seen the state(default value in dictionary is 0)
        return self.count[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        lActions = self.getLegalActions(state)
        # If there is no legal actions, we are in a terminal state
        if len(lActions) == 0:
            return 0
        result = -sys.maxint - 1
        # Find the biggest Q value for the state
        for action in lActions:
            result = max(result, self.getQValue(state, action))
        return result

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"


        lActions = self.getLegalActions(state)
        # If there is no legal actions, we are in a terminal state
        if len(lActions) == 0:
            return None
        # Memorize optimal actions.
        # This is a list because there might be multiple optimal actions
        # with the same values
        resActions = []
        resVal = -sys.maxint - 1
        # For every legal action...
        for action in lActions:
            # Get Q value
            cur = self.getQValue(state, action)
            # If current value is bigger than memorized one,
            # memorize it and put the action in the list after clearing it.
            if cur > resVal:
                resVal = cur
                resActions = []
                resActions.append(action)
            # If the current value is equal to the memorized one, memorize
            # this action too.
            elif cur == resVal:
                resActions.append(action)
        # Return random action chosen from every possible optimal actions
        return random.choice(resActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # If there is no legal actions, we are in a terminal state.
        # If we are in a terminal state, just skip calculations and return None.

        # If we aren't in a terminal state...
        if not len(legalActions) == 0:
            # With a probability of epsilon, return random legal action
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            # With a probability of (1 - epsilon), return optimal legal action
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qVal = self.getQValue(state, action)
        # Update Q values by Q-learning
        self.count[(state, action)] = (1 - self.alpha) * qVal \
                                      + self.alpha * (reward + self.discount * self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        sum = 0.0
        features = self.featExtractor.getFeatures(state, action)
        for key in features:
            # print "feature ", key, " ", features[key]
            sum += features[key] * self.weights[key]

        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # get max q value
        maxQValue = float('-inf')
        nextActions = self.getLegalActions(nextState)
        if len(nextActions) == 0:
            maxQValue = 0.0
        else:
            for nextAction in nextActions:
                qValue = self.getQValue(nextState, nextAction)
                if qValue > maxQValue:
                    maxQValue = qValue

        difference = (reward + (self.discount * maxQValue)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for key in features:
            self.weights[key] += self.alpha * difference * features[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print "Weights:"
            for key in self.weights:
                print key, ":", self.weights[key]
            pass
