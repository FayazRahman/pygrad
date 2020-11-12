import time

import numpy as np


class Op(object):
    def __init__(self):
        pass

    def f(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        return Node(self, input_nodes=inputs)

    def __repr__(self):
        return self.__class__.__name__


class Node(object):
    def __init__(self, op, input_nodes=[]):
        self.op = op
        self.input_nodes = input_nodes
        self.gradient = 0

    def evaluate(self):
        return self.op.f([node.evaluate() for node in self.input_nodes])

    def set_derivative(self, gradient):
        self.gradient += gradient

    def get_derivative(self):
        return self.gradient

    def __repr__(self):
        return "Node"


class PlaceHolder(Node):
    def __init__(self, value=None):
        self.gradient = 0
        if value:
            self.value = value

    def set_value(self, value):
        self.value = value

    def set_derivative(self, gradient):
        self.gradient += gradient

    def evaluate(self):
        return self.value

    def __repr__(self):
        return "PlaceHolder"


class Graph(Op):
    def __init__(self, input_nodes, output_node):
        for node in input_nodes:
            assert isinstance(node, PlaceHolder)
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.input_layer = []
        self.ops = []

    def checkDisconnectedGraph(self):
        # TODO
        pass

    def get_params(self, node):
        params = [
            (input_node.op, (input_node.input_nodes))
            for input_node in node.input_nodes
            if not isinstance(input_node, PlaceHolder)
        ]
        return params

    def condense(self, node):
        got_input = False
        if not isinstance(node, PlaceHolder):
            for input_node in node.input_nodes:
                if not isinstance(input_node, PlaceHolder):
                    if all(
                        [isinstance(i, PlaceHolder) for i in input_node.input_nodes]
                    ):
                        self.input_layer.append(
                            (input_node.op, input_node.input_nodes)
                        )
                        got_input = True
                    else:
                        self.condense(input_node)
            if not got_input:
                self.ops.append(self.get_params(node))
            if node == self.output_node:
                self.ops.append([(node.op, node.input_nodes)])
                self.ops.insert(0, self.input_layer)
            return self.ops

    def f(self, inputs):
        assert len(inputs) == len(
            self.input_nodes
        ), "Expected {} inputs, got {}.".format(len(self.input_nodes), len(inputs))
        for (inpt, input_node) in zip(inputs, self.input_nodes):
            input_node.set_value(inpt)
        next_inputs = []
        for layer in self.ops:
            layer_out = []
            used = 0
            for op in layer:
                if all([isinstance(inpt, PlaceHolder) for inpt in op[1]]):
                    layer_out.append(op[0].f([inpt.evaluate() for inpt in op[1]]))
                else:
                    num_nodes = len(op[1]) - sum(
                        [isinstance(i, PlaceHolder) for i in op[1]]
                    )
                    node_inputs = next_inputs[used : used + num_nodes]
                    op_inputs = []
                    n = 0
                    for inpt in op[1]:
                        if isinstance(inpt, PlaceHolder):
                            op_inputs.append(inpt.evaluate())
                        else:
                            op_inputs.append(node_inputs[n])
                            n += 1
                    layer_out.append(op[0].f(op_inputs))
                    used += num_nodes
            next_inputs = layer_out
        return layer_out[0]

    def backprop(self, node, d):
        d = np.tile(node.op.derivative(d), (len(node.input_nodes),))
        for i, input_node in enumerate(node.input_nodes):
            if isinstance(input_node, PlaceHolder):
                gradient = d[i]
            else:
                gradient = input_node.op.derivative(d[i])
            input_node.set_derivative(gradient)
            if not isinstance(input_node, PlaceHolder):
                self.backprop(input_node, gradient)


class Add(Op):
    def f(self, inputs):
        self.inputs = inputs
        ret = inputs[0]
        for i in range(1, len(inputs)):
            ret += inputs[i]
        return ret

    def derivative(self, d):
        gradient = np.array([1.0])
        return d * (np.tile(gradient, (len(self.inputs),)))


class Multiply(Op):
    def f(self, inputs):
        self.inputs = inputs
        ret = inputs[0]
        for i in range(1, len(inputs)):
            ret *= inputs[i]
        return ret

    def derivative(self, d):
        gradient = np.tile(
            np.prod(self.inputs, axis=0, keepdims=True), len(self.inputs)
        )
        gradient = d * (gradient / self.inputs)
        return gradient


class Subtract(Op):
    def f(self, inputs):
        self.inputs = inputs
        return inputs[0] - inputs[1]

    def derivative(self):
        return np.array([1.0, -1.0])


class Divide(Op):
    def f(self, inputs):
        self.inputs = inputs
        return inputs[0] / inputs[1]

    def derivative(self):
        return


class Pow(Op):
    def f(self, inputs):
        self.inputs = inputs
        self.out = inputs[0] ** inputs[1]
        return self.out

    def derivative(self, d):
        x = self.inputs[0]
        y = self.inputs[1]
        return d * np.array([y * (x ** (y - 1)), self.out * np.log(x)])
"""
Example:
c1 = PlaceHolder()
c2 = PlaceHolder()
c3 = PlaceHolder()

x = Add()([c1, c2])

y = Add()([x, x])

graph = Graph([c1, c2], y)
graph.condense(y)

print(graph.f([1, 2]))
graph.backprop(y, 1)
print(x.gradient)
"""
