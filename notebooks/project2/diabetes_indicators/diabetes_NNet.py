import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Models
# ---------------------------

class Model2L(nn.Module):
    def __init__(self, input_size):
        super(Model2L, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class Model3L(nn.Module):
    def __init__(self, input_size):
        super(Model3L, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Model4L(nn.Module):
    def __init__(self, input_size):
        super(Model4L, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Training + Evaluation
# ---------------------------

def train_and_evaluate(model, X_train, y_train, X_test, y_test,
                       epochs=50, lr=0.01, batch_size=32):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train) > 0.5).float()
        test_preds = (model(X_test) > 0.5).float()

        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

    return train_acc, test_acc


# ---------------------------
# Experiment Functions
# ---------------------------

def run_model_comparison(X_train, y_train, X_test, y_test):
    input_size = X_train.shape[1]

    results = []

    model2 = Model2L(input_size)
    results.append(("2L", *train_and_evaluate(model2, X_train, y_train, X_test, y_test)))

    model3 = Model3L(input_size)
    results.append(("3L", *train_and_evaluate(model3, X_train, y_train, X_test, y_test)))

    model4 = Model4L(input_size)
    results.append(("4L", *train_and_evaluate(model4, X_train, y_train, X_test, y_test)))

    return results


def tune_learning_rate(X_train, y_train, X_test, y_test):
    input_size = X_train.shape[1]
    lrs = [0.1, 0.01, 0.001]

    results = []

    for lr in lrs:
        model = Model3L(input_size)
        train_acc, test_acc = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, lr=lr
        )
        results.append((lr, train_acc, test_acc))

    return results


# ---------------------------
# Print Stuff
# ---------------------------

def print_model_results(results):
    print("\nModel Comparison:")
    print("Model | Train Acc | Test Acc")
    print("-----------------------------")

    for name, train, test in results:
        print(f"{name}    | {train:.4f}     | {test:.4f}")


def print_lr_results(results):
    print("\nLearning Rate Tuning (3L Model):")
    print("LR    | Train Acc | Test Acc")
    print("-----------------------------")

    for lr, train, test in results:
        print(f"{lr} | {train:.4f}     | {test:.4f}")


# ---------------------------
# Main
# ---------------------------

def main():
    # Load dataset
    df = pd.read_csv("diabetes.csv")

    target = "Diabetes_binary"

    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Run experiments
    model_results = run_model_comparison(X_train, y_train, X_test, y_test)
    print_model_results(model_results)

    lr_results = tune_learning_rate(X_train, y_train, X_test, y_test)
    print_lr_results(lr_results)


# ---------------------------
# Main Method Call
# ---------------------------

if __name__ == "__main__":
    main()
