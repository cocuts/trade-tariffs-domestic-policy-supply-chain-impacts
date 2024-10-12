import torch
from .constants import NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS, NUM_PERIODS

def generate_dummy_data():
    return {
        'input_costs': torch.rand((NUM_PERIODS, NUM_GOODS)),
        'Y': torch.rand((NUM_PERIODS, NUM_COUNTRIES, NUM_GOODS)),
        'P': torch.rand((NUM_PERIODS, NUM_COUNTRIES, NUM_GOODS))
    }