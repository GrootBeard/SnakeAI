import numpy as np
import pickle


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
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return biases, weights

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            # print("shape w: {}, shape a: {}".format(np.shape(w), np.shape(a)))
            # print("shape dot(w,a): {}, shape b: {}, shape of sum: {}".format(np.shape(np.dot(w, a)), np.shape(b),
            #                                                                  np.shape(np.dot(w, a) + b)))
            a = sigmoid(np.dot(w, a) + b)
            # print("a: {}".format(a))
        return a

    def save_net(self, file_name):
        data = {"sizes": self.sizes,
                "weights": self.weights,
                "biases": self.biases}
        pickle_out = open(file_name, "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()

    def crossover(self, other):
        return self
