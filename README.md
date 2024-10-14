# Trade, Tariffs, Domestic Policy, and Supply Chain Impacts

This repository contains a Python-based economic model that simulates the effects of trade policies, tariffs, and domestic policies on international trade and supply chains via MCMC with pyro.

## Project Overview

This model aims to analyze how various factors influence international trade dynamics, including:

- Comparative advantage of countries in producing different goods
- Tariffs and trade policies
- Domestic policies, wherein a government can either allow or not allow firm type changes
- Firm-level decisions on production, exports, and inventory management

The model uses PyTorch for tensor operations and Pyro for probabilistic programming and inference.

## Repository Structure

```
economic_model/
│
├── src/
│   └── economic_model/
│       ├── __init__.py
│       ├── constants.py
│       ├── models.py
│       ├── functions.py
│       └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_functions.py
│   └── test_utils.py
│
├── main.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/cocuts/trade-tariffs-domestic-policy-supply-chain-impacts.git
   cd trade-tariffs-domestic-policy-supply-chain-impacts
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Usage

To run the main simulation:

```
python main.py
```

This will execute the economic model and output the results.

## Testing

To run the tests:

```
pytest tests/
```

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## Contact

Cory - corycutsail@gmail.com