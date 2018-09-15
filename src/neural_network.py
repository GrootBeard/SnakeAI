import numpy as np
import pickle


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

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

#net = Network([24, 18, 4])
#net.save_net("test.net")

