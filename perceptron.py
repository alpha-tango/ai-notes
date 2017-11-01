import random

class Perceptron:
    """
    Implementation of a Perceptron.
    Loosely based on
    http://natureofcode.com/book/chapter-10-neural-networks/#chapter10_figure3
    """

    def __init__(self, input_list, learning_constant=0.01):
        self.inputs = input_list
        self.weights = initial_weights()
        self.biased_inputs = biased_inputs()
        self.learning_constant = 0.01

    def _biased_inputs(self):
        """
        Always add a bias input to avoid weirdness
        when all other inputs are 0.
        """
        return self.inputs.append(1.0)

    def _initial_weights(self):
        """
        Begin by assigning all inputs, including the bias,
        random weights.
        """
        return [random.random(-1, 1) for i in len(self.biased_inputs)]

    def _output(self):
        """
        Sum the weighted inputs.
        """
        output = 0
        for i in self.biased_inputs:
            output += i * self.weights[i]
        return output

    def activated(self):
        """
        Decide whether the perceptron "fires" or not.
        """
        if self.output() > 0:
            return 1
        else:
            return 0

    def _update_weights(self, actual):
        """
        Adjust weights towards desired result
        """

        new_weights = []
        for i in self.biased_inputs:
            update = self.error() * i * self.learning_constant
            new_weights.append(self.weights[i] + update)
        self.weights = new_weights
        return new_weights

    def _error(self, actual):
        return actual - self.activated()

    def train(self, actual):
        """
        Learn weights until the correct result is obtained.
        """
        while self.error() != 0:
            self.update_weights(actual)

        return self.weights
