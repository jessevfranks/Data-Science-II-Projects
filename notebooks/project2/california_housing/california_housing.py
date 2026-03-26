import torch
import pandas as pd

from project2.project2_utils import PyTorchNetUtils


def run():
    df = pd.read_csv("../../../data/project2/california_housing_cleaned.csv")
    X_data = df.iloc[:,:-1].to_numpy()
    y_data = df.iloc[:,-1].to_numpy()

    utils = PyTorchNetUtils(X_data, y_data)

    # Run 2L network
    utils.runNeuralNet2L(eta=0.05, bSize=16, f=torch.relu)

    # Run 3L network with specific hidden nodes
    utils.runNeuralNet3L(eta=0.01, bSize=8, nz=10, f1=torch.tanh, f2=torch.relu)

    # Run 4L network
    utils.runNeuralNet4L(eta=0.005, bSize=10, nz1=8, nz2=4, f1=torch.relu, f2=torch.relu, f3=torch.sigmoid)
    return



if __name__ == "__main__":
    run()