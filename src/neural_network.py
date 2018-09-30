import numpy as np
import pickle
import copy


def sigmoid(z):
    return 1.0 / (np.exp(-z) + 1.0)


def load_net(file_name):
    pickle_in = open(file_name, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        (self.biases, self.weights) = self.large_weight_initializer()

    def default_weight_initializer(self):
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return biases, weights

    def large_weight_initializer(self):
        # biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        biases = [np.ones((y, 1)) for y in self.sizes[1:]]
        return biases, weights

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def save_net(self, file_name):
        data = {"sizes": self.sizes,
                "weights": self.weights,
                "biases": self.biases}
        pickle_out = open(file_name, "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def crossover(self, other):
        result = copy.copy(self)
        for i in range(len(self.weights)):
            split = np.random.uniform(0, 1)
            mask = np.random.choice([0, 1], np.shape(self.weights[i]), p=[split, 1-split])
            mask_complement = np.ones(np.shape(self.weights[i])) - mask
            new_weights = mask * self.weights[i] + mask_complement * other.weights[i]
            result.weights[i] = new_weights
        return result

    def mutate(self, rate):
        self.weights = [
            w + np.random.normal(size=np.shape(w)) * np.random.choice([0, 1], (np.shape(w)), p=[1-rate, rate]) / 10
            for w in self.weights]
        # self.biases = [
        #     b + 0.01 * np.random.normal(size=np.shape(b)) * np.random.choice([0, 1], np.shape(b), p=[1-rate, rate])
        #     for b in self.biases]
