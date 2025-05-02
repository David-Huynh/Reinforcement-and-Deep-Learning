"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 3
B. Chan
"""


import numpy as np


class ReLULayer:
    """
    A fully-connected (dense) layer followed by a ReLU activation
    """

    def __init__(self, num_inputs, num_outputs, rng):
        # Initialize weight matrix and bias vector
        self.weights = rng.randn(
            num_outputs,
            num_inputs,
        ) + 1.0

        self.biases = np.zeros((num_outputs, 1))

    def __call__(self, X):
        # Computes ReLU(Wx + b)
        return np.clip(
            np.dot(self.weights, X) + self.biases,
            a_min=0.0,
            a_max=np.inf,
        )


def mse(predictions, targets):
    """
    Computes the mean-squared error (MSE):
    L(y, t) = 0.5 * 1/N sum_{n=1}^N (y_n - t_n)^2,
    where
    1. y_n is the n'th model prediction
    2. t_n is the n'th target
    """

    return 0.5 * np.mean((predictions - targets) ** 2)


class ReLUMLP:
    """
    A multilayer perceptron (MLP) with ReLU activation
    """

    def __init__(self, num_hidden_units, rng):
        self.layers = []
        for num_inputs, num_outputs in zip(
            num_hidden_units[:-1],
            num_hidden_units[1:]
        ):
            self.layers.append(
                ReLULayer(num_inputs, num_outputs, rng)
            )

    def __call__(self, inputs):
        """
        This is used for making predictions
        """
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    def compute_gradients(self, inputs, targets):
        """
        Computes the gradient w.r.t. parameters using backpropagation

        The gradient should be of form:
        {
            0: {
                "weights": np.ndarray,
                "biases": np.ndarray,
            },
            1: {
                "weights": np.ndarray,
                "biases": np.ndarray,
            },
            ...
            L - 1: {
                "weights": np.ndarray,
                "biases": np.ndarray,
            }
        }
        where the key i = 0, ..., L - 1 of the first level corresponds to the gradient
        of the parameters in the i'th layer
        """

        grad = {
            layer_i: {
                "weights": None,
                "biases": None
            }
            for layer_i in range(len(self.layers))
        }
        layer_outputs = []
        loss_partials = []

        # ========================================================
        """
        TODO 1. Forward pass
        Compute each layer's output and store it for later use.
        It will be helpful to propagate layer_outputs to be
        [
            x,
            a_1,
            a_2,
            ...,
            a_L
        ],
        where a_i is the output of the i'th hidden layer.
        This should be a linear scan from first layer to the last layer.
        """
        layer_outputs.append(inputs)
        for layer_i in self.layers:
            layer_outputs.append(layer_i(inputs))
            inputs = layer_outputs[-1]

        
        """
        TODO 2. Backward pass
        Compute the gradient w.r.t. each of the weights and biases
        It will be helpful to propagate loss_partials to be
        [
            d Loss / d z_L,
            d Loss / d z_{L - 1},
            d Loss / d z_{L - 2},
            ...,
            d Loss / d z_{1}
        ]
        where z_i is the pre-activation output of the i'th hidden layer.
        This should be a linear scan from last layer to the first layer.
        """
        for layer_i in reversed(range(len(self.layers))):
            if layer_i == len(self.layers) - 1:
                loss_partials.append((layer_outputs[-1] > 0)*(layer_outputs[-1] - targets) / len(targets[0]))
            else:
                loss_partials.append(np.dot(self.layers[layer_i + 1].weights.T, loss_partials[-1]) * (layer_outputs[layer_i + 1] > 0))
            grad[layer_i]["weights"] = np.dot(loss_partials[-1], layer_outputs[layer_i].T)
            grad[layer_i]["biases"] = np.sum(loss_partials[-1], axis=1, keepdims=True)
        # ========================================================

        return grad

def compute_finite_difference(model, inputs, targets):
    """
    Computes the numerical gradient using finite difference.
    """

    grad_approx = dict()
    h = 1e-7

    orig_loss = mse(model(inputs), targets)

    # Compute finite difference w.r.t. each parameter
    for layer_i, layer in enumerate(model.layers):
        grad_approx[layer_i] = dict()
        for param_type in ["weights", "biases"]:
            orig_param = np.copy(getattr(layer, param_type))
            grad_approx[layer_i][param_type] = np.zeros_like(orig_param)

            for idx in np.ndindex(orig_param.shape):
                new_param = np.copy(orig_param)
                new_param[idx] += h
                setattr(layer, param_type, new_param)

                new_loss = mse(model(inputs), targets)

                grad_approx[layer_i][param_type][idx] = (new_loss - orig_loss) / h

            setattr(layer, param_type, orig_param)

    return grad_approx


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)

    model = ReLUMLP([3, 3, 1], rng)

    # Input shape: d x N
    inputs = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1,],
        [0, 0, 1, 1, 0, 0, 1, 1,],
        [0, 1, 0, 1, 0, 1, 0, 1,],
    ])
    targets = np.array([
        [0, 1, 1, 0, 0, 1, 0, 1]
    ])

    print("Numerical gradient: {}".format(compute_finite_difference(model, inputs, targets)))
    print("Backprop gradient: {}".format(model.compute_gradients(inputs, targets)))
    print("The gradients should be nearly identical.")
