"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 3
B. Chan
"""


import _pickle as pickle
import math
import numpy as np
import os
import timeit
import torch


TRAIN_VAL_SPLIT = 0.8
SPLIT_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(batch, model, optimizer):
    """
    Performs a step of batch update.
    """
    
    loss_fn = torch.nn.CrossEntropyLoss()

    preds = model(batch["train_X"])
    targs = batch["train_y"]

    optimizer.zero_grad()
    loss = loss_fn(preds, targs)
    loss.backward()
    optimizer.step()

    step_info = {
        "loss": loss.detach().cpu().item(),
    }

    # Compute gradient norm
    for n, p in filter(
        lambda layer: layer[1].grad is not None,
        model.named_parameters()
    ):
        step_info["grad_norm/{}".format(n)] = p.grad.data.norm(2).item()
        step_info["param_norm/{}".format(n)] = p.norm(2).item()

    return step_info


def train(dataset, model, optimizer, args, run_name):
    """
    Runs the training loop.
    """

    # Split dataset into training and validation
    num_train = math.floor(len(dataset) * TRAIN_VAL_SPLIT)
    num_val = len(dataset) - num_train

    # Make sure to split the same way
    gen = torch.Generator()
    gen.manual_seed(SPLIT_SEED)

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [num_train, num_val],
        generator=gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    best_val_loss = np.inf

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(run_name, "best.pt")
    )

    logs = {
        "train": dict(),
        "validation": dict(),
    }
    for epoch_i in range(args.num_epochs):
        # Train
        tic = timeit.default_timer()
        step_infos = {}
        for (train_X, train_y) in train_loader:
            step_info = train_step(
                {
                    "train_X": train_X.to(DEVICE),
                    "train_y": train_y.to(DEVICE),
                },
                model,
                optimizer,
            )
            for k, v in step_info.items():
                step_infos.setdefault(k, [])
                step_infos[k].append(v)

        step_infos = {k: np.mean(v) for k, v in step_infos.items()}
        toc = timeit.default_timer()

        print("Epoch {} ==========================================".format(epoch_i))
        print("Time taken for training: {:4f}s".format(toc - tic))
        for k, v in step_info.items():
            logs["train"].setdefault(k, [])
            logs["train"][k].append(v)
            print("Avg {}: {:4f}".format(k, v))

        # Validation
        val_metrics = evaluate(val_set, model, args)
        curr_val_loss = val_metrics["loss"]

        for k, v in val_metrics.items():
            logs["validation"].setdefault(k, [])
            logs["validation"][k].append(v)
            print("Avg {}: {}".format(k, v))

        if best_val_loss > curr_val_loss:
            print("Currently model with lowest validation: {:4f}".format(
                curr_val_loss
            ))
            best_val_loss = curr_val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(run_name, "best.pt")
            )

        pickle.dump(logs, open(os.path.join(run_name, "logs.pkl"), "wb"))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(run_name, "final.pt")
    )


def evaluate(dataset, model, args):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    tic = timeit.default_timer()
    metrics = {
        "loss": [],
        "accuracy": [],
    }
    model.eval()
    for (X, y) in dataloader:
        batch = {
            "X": X.to(DEVICE),
            "y": y.to(DEVICE),
        }
        
        with torch.no_grad():
            loss_fn = torch.nn.CrossEntropyLoss()

            preds = model(batch["X"])
            targs = batch["y"]

            loss = loss_fn(preds, targs)
            accuracy = torch.mean(
                (torch.argmax(preds, dim=-1) == targs).float()
            )

        metrics["loss"].append(loss.detach().cpu().item())
        metrics["accuracy"].append(accuracy.detach().cpu().item())
    model.train()
    toc = timeit.default_timer()
    print("Time taken for evaluation: {:4f}s".format(toc - tic))

    metrics = {k: np.mean(v) for k, v in metrics.items()}

    return metrics
