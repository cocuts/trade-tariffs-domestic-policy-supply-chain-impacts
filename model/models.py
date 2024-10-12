import torch
import pyro
import pyro.distributions as dist
from pyro import plate
import numpy as np

from .constants import NUM_COUNTRIES, NUM_FIRMS, NUM_PERIODS, NUM_GOODS
from .functions import (
    production_function, inventory_cost, demand_function, firm_profit,
    initialize_tariffs, update_tariffs, determine_comparative_advantage,
    can_change_type, cournot_equilibrium, calculate_potential_profit,
    initialize_firm_types
)

def model(data):
    # Prior distributions for model parameters
    A = pyro.sample('A', dist.Gamma(torch.ones((NUM_COUNTRIES, NUM_GOODS)), torch.ones((NUM_COUNTRIES, NUM_GOODS))))
    alpha = pyro.sample('alpha', dist.Dirichlet(torch.ones(NUM_GOODS)))
    h = pyro.sample('h', dist.Gamma(2, 2))  # Inventory holding cost parameter
    fixed_cost = pyro.sample('fixed_cost', dist.Gamma(10, 1))
    comparative_advantage = determine_comparative_advantage(A)
    a = pyro.sample('a', dist.Normal(100, 10).expand([NUM_COUNTRIES, NUM_GOODS]))
    b = pyro.sample('b', dist.Gamma(2, 0.1).expand([NUM_COUNTRIES, NUM_GOODS]))
    # Initialize Country types
    country_types = np.array([False,False,False,False,False])
    
    # Initialize firm types
    firm_types = initialize_firm_types(NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS)
    
    # Initialize inventories
    I = torch.zeros((NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS))
    
    # Initialize tariffs
    tariffs = initialize_tariffs(NUM_COUNTRIES, NUM_GOODS)

    # Add temperature parameter
    temperature = pyro.param("temperature", torch.tensor(1.0), constraint=dist.constraints.positive)

    # Simulation loop
    for t in range(NUM_PERIODS):
        # Update tariffs each period
        tariffs = update_tariffs(tariffs, t)
        change_allowed = can_change_type(t, country_types)


        # Introduce trade policy shock at t=10
        if t == 10:
            tariffs = pyro.sample(f'tariffs_{t}', dist.Uniform(0, 0.5).expand(tariffs.shape))

        if t == 20:
            country_types = np.array([True, False, False, False, False])
        
        # Production decisions (Cournot equilibrium)
        Y = torch.zeros((NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS))
        X = torch.zeros((NUM_COUNTRIES, NUM_COUNTRIES, NUM_FIRMS, NUM_GOODS))
        for g in range(NUM_GOODS):
            for k in range(NUM_COUNTRIES):
                mc = data['input_costs'][t, g] / A[k, g]  # Marginal cost
                Q, P = cournot_equilibrium(a[k, g], b[k, g], mc, NUM_FIRMS)
                Y[k, :, g] = Q / NUM_FIRMS  # Equal split among firms (simplified)
        
        # Export decisions (simplified)
        for k in range(NUM_COUNTRIES):
            for j in range(NUM_COUNTRIES):
                if k != j:
                    X[k, j] = pyro.sample(f'X_{k}_{j}_{t}', dist.Uniform(0, Y[k] * 0.5))
        
        # Calculate prices
        P = torch.stack([demand_function(a[:, g], b[:, g], Y[:, :, g], X[:, :, :, g]) for g in range(NUM_GOODS)]).T
        # Inventory decisions
        for k in range(NUM_COUNTRIES):
            for f in range(NUM_FIRMS):
                I[k, f] = pyro.sample(f'I_{k}_{f}_{t}', dist.Uniform(0, Y[k, f] * 0.5))
        
        # Calculate profits
        S = Y - I  # Sales (num_countries, num_firms, num_goods)
        prod_costs = Y * data['input_costs'][t].unsqueeze(0).unsqueeze(0)  # (num_countries, num_firms, num_goods)
        inv_costs = inventory_cost(I, h)  # (num_countries, num_firms, num_goods)
        change_type = torch.zeros((NUM_COUNTRIES, NUM_FIRMS))  # No type changes by default
        
        profits = firm_profit(P, S, X, prod_costs, inv_costs, fixed_cost, change_type)

        change_allowed = can_change_type(t, comparative_advantage)
        for k in range(NUM_COUNTRIES):
            if change_allowed[k]:
                avg_profit = torch.mean(profits[k])
                with plate(f'firms_{k}_{t}', NUM_FIRMS):
                    change_probs = pyro.sample(f'change_prob_{k}_{t}', dist.Beta(2, 2))
                    change_decisions = torch.Tensor([1 if i > 0.7 else 0 for i in change_probs])
                    #pyro.sample(f'change_decision_{k}_{t}', dist.Bernoulli(change_probs))

                for f in range(NUM_FIRMS):
                    # print('Change Probs', change_probs)
                    # print('Change Decisions', change_decisions)
                    if profits[k, f] < avg_profit and change_decisions[f].item():
                        current_type = firm_types[k, f].item()
                        potential_profits = []

                        for new_type_candidate in range(NUM_GOODS):
                            if new_type_candidate != current_type:
                                simulated_profit = calculate_potential_profit(
                                    k, f, new_type_candidate, A, alpha, h, fixed_cost, 
                                    comparative_advantage[k], tariffs, P, Y, X, data['input_costs'][t],
                                    firm_types
                                )
                                potential_profits.append(simulated_profit - fixed_cost)
                            else:
                                potential_profits.append(profits[k, f])

                        profit_diffs = torch.tensor(potential_profits) - profits[k, f]
                        type_probs = torch.softmax(profit_diffs / temperature, dim=0)
                        print('type probs:',type_probs)
                        new_type_tensor = pyro.sample(f'new_type_{k}_{f}_{t}', dist.Categorical(type_probs))
                        print('New Type Tensor',new_type_tensor)
                        if new_type_tensor.dim() > 0:
                            new_type = new_type_tensor[torch.argmax(new_type_tensor)].flatten()
                        else: 
                            new_type = new_type_tensor.item()

                        if new_type != current_type:
                            firm_types[k, f] = new_type
                            profits[k, f] = potential_profits[new_type]

        # Observe data
        pyro.sample(f'obs_Y_{t}', dist.Normal(Y.sum(dim=1), 0.1), obs=data['Y'][t])
        pyro.sample(f'obs_P_{t}', dist.Normal(P, 0.1), obs=data['P'][t])
    
    return Y, X, I, P, firm_types, profits, tariffs

def infer_parameters(data, num_samples=1000):
    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=200)
    mcmc.run(data)
    return mcmc.get_samples()