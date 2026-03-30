import torch
import pandas as pd

from project2.project2_utils import PyTorchNetUtils

def run():
    df = pd.read_csv("../../../data/project2/california_housing_cleaned.csv")

    X_data = df.drop(columns=['median_house_value']).to_numpy()
    y_data = df['median_house_value'].to_numpy()

    utils = PyTorchNetUtils(X_data, y_data)

    utils.runNeuralNet2L(eta=0.001, bSize=16)
    utils.runNeuralNet3L(eta=0.001, bSize=32, nz=20)
    utils.runNeuralNet4L(eta=0.001, bSize=64)

    # Pick an Architecture (Deep Model)
    utils.runNeuralNetDeep(eta=0.001, bSize=64, nz1=64, nz2=32, epochs=100)

    print("HYPER-PARAMETER EXPERIMENTATION")

    etas = [0.1, 0.01, 0.001]
    batch_sizes = [16, 64]

    for lr in etas:
        for bs in batch_sizes:
            print(f"\n>>> TESTING: Learning Rate={lr}, Batch Size={bs}")
            # Testing on 3L as a representative model
            utils.runNeuralNet3L(eta=lr, bSize=bs, nz=15, epochs=50)

if __name__ == "__main__":
    run()