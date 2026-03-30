import torch
import pandas as pd

from project2.project2_utils import PyTorchNetUtils, NeuralNet_3L


def run():
    df = pd.read_csv("../../../data/project2/california_housing_cleaned.csv")
    X_data = df.iloc[:,:-1].to_numpy()
    y_data = df.iloc[:,-1].to_numpy()

    utils = PyTorchNetUtils(X_data, y_data)

    utils.runNeuralNet2L(eta=0.05, bSize=16, f=torch.relu)
    utils.runNeuralNet3L(eta=0.01, bSize=8, nz=10, f1=torch.tanh, f2=torch.relu)
    utils.runNeuralNet4L(eta=0.005, bSize=10, nz1=8, nz2=4, f1=torch.relu, f2=torch.relu, f3=torch.sigmoid)
    utils.runNeuralNetDeep(eta=0.001, bSize=64, nz1=64, nz2=32)

    print("\n=== STARTING HYPER-PARAMETER EXPERIMENTATION ===")

    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [16, 64]
    hidden_configs = [5, 20] # nz for 3L network

    for lr in learning_rates:
        for bs in batch_sizes:
            for nz in hidden_configs:
                print(f"\nTesting: lr={lr}, batch={bs}, nz={nz}")
                model = NeuralNet_3L(utils.input_dim, nz, utils.output_dim, torch.relu, torch.relu)
                # Note: This will print results to console based on your existing _train_model_tts
                utils._train_model_tts(model, eta=lr, bSize=bs, epochs=50)

    return



if __name__ == "__main__":
    run()