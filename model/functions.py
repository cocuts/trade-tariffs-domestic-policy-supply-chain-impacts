import torch
import pyro
import pyro.distributions as dist
from .constants import NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS

def production_function(A, inputs, alpha):
    return A * torch.prod(inputs ** alpha, dim=-1)

def inventory_cost(I, h):
    return h * I

def demand_function(a, b, Y, X):
    Q = Y.sum(dim=1) + X.sum(dim=(0, 2))  # Total quantity in market
    return a - b * Q

def firm_profit(P, S, X, prod_costs, inv_costs, fixed_cost, change_type):
    assert P.shape == (NUM_COUNTRIES, NUM_GOODS)
    assert S.shape == (NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    assert X.shape == (NUM_COUNTRIES, NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    assert prod_costs.shape == (NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    assert inv_costs.shape == (NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    assert change_type.shape == (NUM_COUNTRIES, NUM_FIRMS)
    
    domestic_revenue = torch.sum(P.unsqueeze(1) * S, dim=-1)
    export_revenue = torch.sum(P.unsqueeze(1).unsqueeze(1) * X, dim=(1, 3))
    total_revenue = domestic_revenue + export_revenue
    total_costs = torch.sum(prod_costs + inv_costs, dim=-1) + fixed_cost * change_type
    profit = total_revenue - total_costs
    
    assert profit.shape == (NUM_COUNTRIES, NUM_FIRMS)
    return profit

def initialize_tariffs(num_countries, num_goods):
    """Initialize tariffs with low initial values."""
    return torch.rand((num_countries, num_countries, num_goods)) * 0.05  # 0-5% initial tariffs

def update_tariffs(current_tariffs, t, shock_probability=0.1, max_change=0.02):
    """
    Update tariffs based on time and random shocks.
    """
    assert current_tariffs.shape == (NUM_COUNTRIES, NUM_COUNTRIES, NUM_GOODS), "Invalid tariffs shape"
    
    new_tariffs = current_tariffs.clone()
    
    shock = torch.rand(current_tariffs.shape) < shock_probability
    change = (torch.rand(current_tariffs.shape) * 2 - 1) * max_change
    new_tariffs += shock.float() * change
    
    new_tariffs = torch.clamp(new_tariffs, 0, 1)
    
    assert new_tariffs.shape == current_tariffs.shape, "Tariff shape changed unexpectedly"
    return new_tariffs

def determine_comparative_advantage(A):
    """
    Determines which goods each country has a comparative advantage in.
    """
    assert A.shape == (NUM_COUNTRIES, NUM_GOODS), "Invalid A shape"
    
    # Calculate the relative productivity for each country-good pair
    relative_productivity = A / A.mean(dim=0)
    
    # Determine the good with the highest relative productivity for each country
    comparative_advantage = torch.argmax(relative_productivity, dim=1)
    
    assert comparative_advantage.shape == (NUM_COUNTRIES,), "Invalid comparative advantage shape"
    return comparative_advantage

def can_change_type(t, country_types, policy_change_interval=10):
    """
    Determines whether firms in each country can change their type at time t.
    
    :param t: Current time period
    :param country_types: Boolean tensor where True indicates GMS and False indicates CMD
    :param policy_change_interval: Interval at which GMS countries allow type changes
    """
    assert country_types.shape == (NUM_COUNTRIES,), "Invalid country_types shape"
    
    # CMD countries (False in country_types) can always change
    print('Country Types', country_types)
    cmd_countries = country_types == False

    # GMS countries (True in country_types) can only change at intervals
    gms_countries = country_types & (t % policy_change_interval == 0)
    
    change_allowed = cmd_countries | gms_countries
    
    assert change_allowed.shape == (NUM_COUNTRIES,), "Invalid change_allowed shape"
    return change_allowed

def cournot_equilibrium(a, b, mc, n_firms):
    """
    Calculate the Cournot equilibrium quantities and prices.
    """
    assert isinstance(n_firms, int), "n_firms must be an integer"
    
    Q = (a - mc) / (b * (n_firms + 1))
    P = a - b * Q * n_firms
    
    assert Q.shape == P.shape, "Q and P should have the same shape"
    return Q, P

def calculate_potential_profit(k, f, new_type, A, alpha, h, fixed_cost, comparative_advantage, tariffs, P, Y, X, input_costs, firm_types):
    """
    Calculate the potential profit for a firm if it switches to a new type.
    """
    assert 0 <= k < NUM_COUNTRIES, f"Invalid country index: {k}"
    assert 0 <= f < NUM_FIRMS, f"Invalid firm index: {f}"
    assert 0 <= new_type < NUM_GOODS, f"Invalid new type: {new_type}"
    assert A.shape == (NUM_COUNTRIES, NUM_GOODS), "Invalid A shape"
    assert alpha.shape == (NUM_GOODS,), "Invalid alpha shape"
    assert tariffs.shape == (NUM_COUNTRIES, NUM_COUNTRIES, NUM_GOODS), "Invalid tariffs shape"
    assert P.shape == (NUM_COUNTRIES, NUM_GOODS), "Invalid P shape"
    assert Y.shape == (NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS), "Invalid Y shape"
    assert X.shape == (NUM_COUNTRIES, NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS), "Invalid X shape"
    assert input_costs.shape == (NUM_GOODS,), "Invalid input_costs shape"
    assert firm_types.shape == (NUM_COUNTRIES, NUM_FIRMS), "Invalid firm_types shape"
    
    new_productivity = A[k, new_type]
    estimated_Y = Y[k, f].clone()
    estimated_Y[new_type] *= (new_productivity / A[k, firm_types[k, f]])
    
    estimated_X = X[k, :, f].clone()
    estimated_X[:, new_type] *= (new_productivity / A[k, firm_types[k, f]])
    
    new_prod_costs = estimated_Y * input_costs[new_type]
    
    estimated_I = torch.min(estimated_Y * 0.1, Y[k, f] * 0.1)
    new_inv_costs = h * estimated_I
    
    domestic_revenue = P[k, new_type] * (estimated_Y[new_type] - estimated_I[new_type])
    export_revenue = torch.sum(P[:, new_type] * estimated_X[:, new_type])
    total_revenue = domestic_revenue + export_revenue
    
    tariff_costs = torch.sum(tariffs[:, k, new_type] * P[:, new_type] * estimated_X[:, new_type])
    
    total_costs = torch.sum(new_prod_costs) + torch.sum(new_inv_costs) + fixed_cost
    
    profit = total_revenue - total_costs - tariff_costs
    
    profit = total_revenue - total_costs - tariff_costs
    
    # Apply comparative advantage boost if the new type matches the country's advantage
    if new_type == comparative_advantage:
        profit *= 1.1  # 10% boost for comparative advantage
    
    return profit.item()

def initialize_firm_types(num_countries, num_firms, num_goods):
    """
    Initialize firm types randomly.
    """
    return torch.randint(0, num_goods, (num_countries, num_firms))