"""

Classes used for training neural nets with numpy.

"""

from math_fns import sigmoid, sigmoid_prime
import numpy as np
from typing import Any, Callable, Optional


class Op:
    """
    Defines an operation and its gradient.

    Each child class defines a fwd_prop and bck_prop function for forward and back propagation.

    fwd_prop: result of forward propagation
    bck_prop: returns input gradeint * gradient of this op

    Note that for simplicity, we do not have a separate optimizer class for
    updating trainable variables given gradients for now.

    """

    def fwd_prop(self, input: Any) -> np.float:
        raise NotImplementedError

    def bck_prop(self, gradient: Any) -> np.float:
        raise NotImplementedError


class Sigmoid(Op):
    def __init__(self):
        self.input = None

    def fwd_prop(self, input: np.array) -> np.float:
        self.input = input
        return sigmoid(self.input)

    def bck_prop(self, gradient: np.array, *args: Any, **kwargs: Any) -> np.float:
        assert self.input is not None
        return gradient * sigmoid_prime(self.input)


class Dense(Op):
    """
    XW = Z
    """

    def __init__(self, dim_in, dim_out, init_fn=lambda x, y: np.zeros(shape=(x, y))):
        self.input = None
        self.W = init_fn(dim_in, dim_out)
        self.b = init_fn(1, dim_out)

    def fwd_prop(self, input):
        self.input = input
        return np.matmul(self.input, self.W) + self.b

    def bck_prop(self, gradient, learning_rate, *args, **kwargs):
        """
        gradient = dc/dz
        """
        assert self.input is not None
        dc_dx = np.matmul(gradient, np.transpose(self.W))
        dc_dw = np.matmul(np.transpose(self.input), gradient)
        dc_db = gradient
        self.W -= learning_rate * dc_dw
        self.b = self.b - np.mean(learning_rate * dc_db, axis=0)
        return dc_dx


class Graph:
    def __init__(self):
        self.nodes = []

    def __lshift__(self, node):
        self.nodes.append(node)

    def append(self, node):
        self.nodes.append(node)

    def fwd_prop(self, input):
        for node in self.nodes:
            input = node.fwd_prop(input)
        return np.squeeze(input)

    def bck_prop(self, gradient, learning_rate):
        # NB node updates parameters internally
        for node in reversed(self.nodes):
            gradient = node.bck_prop(gradient, learning_rate=learning_rate)


class Trainer:
    def __init__(self, graph: Graph, cost_fn: Callable, cost_grad: Callable):
        self.graph = graph
        self.cost_fn = cost_fn
        self.cost_grad = cost_grad

    def step(self, input, labels, learning_rate, return_cost=False):
        pred = self.graph.fwd_prop(input)
        grad = self.cost_grad(labels, pred)
        self.graph.bck_prop(np.expand_dims(grad, axis=1), learning_rate=learning_rate)
        if return_cost:
            return self.cost_fn(labels, pred)
