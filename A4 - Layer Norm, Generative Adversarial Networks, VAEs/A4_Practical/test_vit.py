        """
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 4
B. Chan
"""


import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.vit import (
    VisionTransformer,
    TransformerEncoder,
)

BATCH_SIZE = 3
EMBED_DIM = 16
NUM_HEADS = 2
NUM_BLOCKS = 3
WIDENING_FACTOR = 4

import torch

def test_encoder_forward():
    model = TransformerEncoder(
        EMBED_DIM,
        NUM_HEADS,
        WIDENING_FACTOR,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "test_vit_model.pt"),
            weights_only=False,
        )["encoder"]
    )
    return model.forward(test_data["x"])

def test_vit_forward_without_cls_token():
    model = VisionTransformer(
        EMBED_DIM,
        NUM_BLOCKS,
        NUM_HEADS,
        WIDENING_FACTOR,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "test_vit_model.pt"),
            weights_only=False,
        )["vit"]
    )
    return model.forward(test_data["images"])

def test_vit_forward_with_cls_token():
    model = VisionTransformer(
        EMBED_DIM,
        NUM_BLOCKS,
        NUM_HEADS,
        WIDENING_FACTOR,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(currentdir, "test_vit_model.pt"),
            weights_only=False,
        )["vit"]
    )
    return model.forward(test_data["x"], test_data["cls_token"])


if __name__ == "__main__":
    test_data = torch.load(
        os.path.join(currentdir, "test_vit_data.pt"),
        weights_only=False,
    )

    encoder_result = test_encoder_forward()
    vit_with_cls_result = test_vit_forward_with_cls_token()
    vit_result = test_vit_forward_without_cls_token()
    
    # print("Test for TransformerEncoder forward")
    # print("Correct: {}".format(
    #     torch.allclose(encoder_result, test_data["encoder_forward"]))
    #      max_error_encoder = torch.max(torch.abs(encoder_result - test_data["encoder_forward"]))

    # )

    # print("Test for VisionTransformer forward with CLS token")
    # print("Correct: {}".format(
    #     torch.allclose(vit_with_cls_result, test_data["vit_forward_with_cls_token"]))
    # )

    # print("Test for VisionTransformer forward without CLS token")
    # print("Correct: {}".format(
    #     torch.allclose(vit_result, test_data["vit_forward_no_cls_token"]))
    # )
    tolerance = 1e-5  # Define the tolerance for maximum error

    print("Test for TransformerEncoder forward")
    max_error_encoder = torch.max(torch.abs(encoder_result - test_data["encoder_forward"]))
    print("Correct:", max_error_encoder.item() < tolerance)

    print("Test for VisionTransformer forward with CLS token")
    max_error_vit_with_cls = torch.max(torch.abs(vit_with_cls_result - test_data["vit_forward_with_cls_token"]))
    print("Correct:", max_error_vit_with_cls.item() < tolerance)

    print("Test for VisionTransformer forward without CLS token")
    max_error_vit_without_cls = torch.max(torch.abs(vit_result - test_data["vit_forward_no_cls_token"]))
    print("Correct:", max_error_vit_without_cls.item() < tolerance)
