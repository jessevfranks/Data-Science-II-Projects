import os
import pandas as pd
from project2.project2_utils import PyTorchNetUtils

# from ..project2_utils import PyTorchNetUtils

# to run this, in terminal cd to 'notebooks' folder
# python -m project2.diabetes_indicators.diabetesMain

def run():
    # Load the 50-50 split diabetes dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'project2', 'diabetes.csv')
    df = pd.read_csv(data_path)
    #df = pd.read_csv("diabetes.csv")
    # Separate features and target
    X_data = df.drop(columns=['Diabetes_binary']).to_numpy()
    y_data = df['Diabetes_binary'].to_numpy()

    # Initialize your utility class
    utils = PyTorchNetUtils(X_data, y_data)

    # Run 2L, 3L, and 4L neural networks
    utils.runNeuralNet2L(eta=0.01, bSize=16, epochs=50)
    utils.runNeuralNet3L(eta=0.01, bSize=32, nz=16, epochs=50)
    utils.runNeuralNet4L(eta=0.01, bSize=64, nz1=32, nz2=16, epochs=50)

    # Hyperparameter experiments for 3L model
    print("\n=== HYPERPARAMETER EXPERIMENTATION ===")
    etas = [0.1, 0.01, 0.001]
    batch_sizes = [16, 64]

    for lr in etas:
        for bs in batch_sizes:
            print(f"\n>>> Testing 3L: LR={lr}, Batch Size={bs}")
            utils.runNeuralNet3L(eta=lr, bSize=bs, nz=16, epochs=50)


if __name__ == "__main__":
    run()