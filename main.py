from model.models import infer_parameters
from model.utils import generate_dummy_data

def main():
    data = generate_dummy_data()
    samples = infer_parameters(data)
    
    for param in ['A', 'alpha', 'h', 'fixed_cost', 'a', 'b']:
        print(f"{param} mean: {samples[param].mean(axis=0)}")
        print(f"{param} std: {samples[param].std(axis=0)}")

if __name__ == "__main__":
    main()