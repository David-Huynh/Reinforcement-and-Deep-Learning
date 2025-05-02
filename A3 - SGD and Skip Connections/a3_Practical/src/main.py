"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 3
B. Chan
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datetime import datetime
from torchvision import datasets
from torchvision import transforms

import _pickle as pickle
import argparse
import json
import numpy as np
import torch
import uuid

import src.model as models

from src.train import train, evaluate


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 10)

    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    """
    Entrypoint for training a model on FashionMNIST.
    """
    set_seed(args.seed)

    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.FashionMNIST(
        os.path.join(args.save_path, "datasets"),
        train=not args.eval,
        download=True,
        transform=transform,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and optimizer
    model = getattr(models, args.model)()
    model.to(device)

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.learning_rate
    )

    # Load checkpoint, if provided
    if args.load_path and os.path.isfile(args.load_path):
        print("Loading parameters from {}".format(args.load_path))
        checkpoint = torch.load(args.load_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Logging-related content
    run_id = str(uuid.uuid4())
    time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
    run_name = os.path.join(
        args.save_path,
        "models",
        "{}-{}".format(time_tag, run_id),
    )

    if args.eval:
        # Evaluation entry
        assert args.load_path is not None and os.path.isfile(args.load_path), (
            "Evaluation must be executed with a saved model"
        )
        eval_metrics = evaluate(dataset, model, args)
        pickle.dump(
            eval_metrics,
            open(
                os.path.join(os.path.dirname(args.load_path), "eval.pkl"),
                "wb",
            )
        )
    else:
        # Training entry
        os.makedirs(
            run_name,
            exist_ok=True,
        )

        json.dump(
            args,
            open(os.path.join(run_name, "config.json"), "w"),
            default=lambda s: vars(s),
        )

        train(dataset, model, optimizer, args, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of times the data is iterated on"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The minibatch size per update"
    )

    parser.add_argument(
        "--model",
        choices=["MLP", "SkipConnectionMLP", "CustomModel"],
        default="MLP",
        help="The model to use"
    )

    parser.add_argument(
        "--optimizer",
        choices=["Adam", "SGD"],
        default="SGD",
        help="The optimizer to use"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="The learning rate"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./logs",
        help="The path to store any artifacts"
    )

    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="The path to restore an artifact"
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether or not to evaluate model"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for randomness"
    )

    args = parser.parse_args()
    main(args)
