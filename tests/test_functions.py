import pytest
import torch
from model.functions import update_tariffs, determine_comparative_advantage, can_change_type, cournot_equilibrium, calculate_potential_profit
from model.constants import NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS

def test_update_tariffs():
    initial_tariffs = torch.rand((NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)) * 0.05
    updated_tariffs = update_tariffs(initial_tariffs, t=1)
    assert updated_tariffs.shape == (NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    assert torch.all(updated_tariffs >= 0) and torch.all(updated_tariffs <= 1)

def test_determine_comparative_advantage():
    A = torch.rand((NUM_COUNTRIES, NUM_GOODS))
    comp_adv = determine_comparative_advantage(A)
    assert comp_adv.shape == (NUM_COUNTRIES,)
    assert comp_adv.dtype == torch.bool

# In tests/test_functions.py

def test_determine_comparative_advantage():
    A = torch.rand((NUM_COUNTRIES, NUM_GOODS))
    comp_adv = determine_comparative_advantage(A)
    assert comp_adv.shape == (NUM_COUNTRIES,)
    assert torch.all(comp_adv >= 0) and torch.all(comp_adv < NUM_GOODS)

def test_can_change_type():
    country_types = torch.randint(0, 2, (NUM_COUNTRIES,), dtype=torch.bool)
    change_allowed_10 = can_change_type(t=10, country_types=country_types)
    change_allowed_11 = can_change_type(t=11, country_types=country_types)
    assert change_allowed_10.shape == (NUM_COUNTRIES,)
    assert change_allowed_10.dtype == torch.bool
    assert torch.all(change_allowed_10 == (~country_types | (country_types & True)))
    assert torch.all(change_allowed_11 == (~country_types))

def test_calculate_potential_profit():
    # ... (setup code)
    comparative_advantage = torch.randint(0, NUM_GOODS, (NUM_COUNTRIES,))
    profit = calculate_potential_profit(
        k=0, f=0, new_type=1, A=A, alpha=alpha, h=h, fixed_cost=fixed_cost,
        comparative_advantage=comparative_advantage, tariffs=tariffs,
        P=P, Y=Y, X=X, input_costs=input_costs, firm_types=firm_types
    )
    assert isinstance(profit, float)