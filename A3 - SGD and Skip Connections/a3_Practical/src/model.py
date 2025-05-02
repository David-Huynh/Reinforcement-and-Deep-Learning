"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 3
B. Chan
"""


import numpy as np
import torch.nn as nn


IMG_DIM = (1, 28, 28)
FLATTENED_IMG_DIM = np.prod(IMG_DIM)
NUM_CLASSES = 10


class MLP(nn.Module):
    """
    A two-layered multilayer perceptron
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=NUM_CLASSES,
            bias=True,
        )

    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        x = self.hidden_1(x)
        x = nn.functional.relu(x)
        x = self.hidden_2(x)
        x = nn.functional.relu(x)
        x = self.out(x)

        return x


class SkipConnectionMLP(nn.Module):
    """
    A two-layered multilayer perceptron with a skip connection
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=FLATTENED_IMG_DIM,
            bias=True,
        )

        self.out = nn.Linear(
            in_features=FLATTENED_IMG_DIM,
            out_features=NUM_CLASSES,
            bias=True,
        )

    def forward(self, x):
        # Flatten the image
        x = x.reshape(len(x), -1)

        out = self.hidden_1(x)
        out = nn.functional.relu(out)
        out = self.hidden_2(out) + x
        out = nn.functional.relu(out)
        out = self.out(out)

        return out


class CustomModel(nn.Module):
    """
    Your custom model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2
        )
        self.hidden_2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
        )
        self.hidden_22 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.hidden_3 = nn.Linear(
            in_features=512,
            out_features=256,
            bias=True,
        )
        self.hidden_33 = nn.Linear(
            in_features=256,
            out_features=16,
            bias=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.hidden_4 = nn.Linear(
            in_features=16,
            out_features=NUM_CLASSES,
            bias=True,
        )
        self.out = nn.Softmax(dim=1)


        
    def forward(self, x):
        # ========================================================
        # TODO: Implement the forward pass of your model
        x = nn.functional.relu(self.bn2(self.hidden_1(x)))

        # Second convolution, batch norm, and ReLU.
        x = nn.functional.relu((self.hidden_2(x)))
        x = nn.functional.relu(self.hidden_22(x))
        x = self.pool(x)
        # Flatten.
        x = self.flatten(x)
        # Fully connected layers with dropout.
        x = nn.functional.relu(self.hidden_3(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.hidden_33(x))
        x = self.dropout(x)

        x = self.hidden_4(x)
        x = self.out(x)
        return x
        # ========================================================
