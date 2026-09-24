import torch

from src.models.model import IncomeClassifier


def test_model_output_shape():

    input_dim = 100

    model = IncomeClassifier(
        input_dim=input_dim
    )

    X = torch.randn(
        8,
        input_dim
    )

    output = model(X)

    assert output.shape == (8, 1)