import pytest
import torch
from model.models import model, infer_parameters
from model.utils import generate_dummy_data

def test_model():
    data = generate_dummy_data()
    Y, X, I, P, firm_types, profits, tariffs = model(data)
    # Add assertions to check the shapes and types of outputs

def test_infer_parameters():
    data = generate_dummy_data()
    samples = infer_parameters(data, num_samples=10)  # Use a small number for quick testing
    # Add assertions to check the shapes and types of samples