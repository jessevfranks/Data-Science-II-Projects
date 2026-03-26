import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


class NeuralNet_2L(nn.Module):
    """2 Layers: Input -> Output"""
    def __init__(self, input_dim, output_dim, f):
        super(NeuralNet_2L, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.activation = f

    def forward(self, x):
        return self.activation(self.layer1(x))

class NeuralNet_3L(nn.Module):
    """3 Layers: Input -> Hidden -> Output"""
    def __init__(self, input_dim, hidden_dim, output_dim, f1, f2):
        super(NeuralNet_3L, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = f1
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = f2

    def forward(self, x):
        out = self.act1(self.layer1(x))
        return self.act2(self.layer2(out))

class NeuralNet_4L(nn.Module):
    """4 Layers: Input -> Hidden 1 -> Hidden 2 -> Output"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, f1, f2, f3):
        super(NeuralNet_4L, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.act1 = f1
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = f2
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.act3 = f3

    def forward(self, x):
        out = self.act1(self.layer1(x))
        out = self.act2(self.layer2(out))
        return self.act3(self.layer3(out))

class PyTorchNetUtils:
    def __init__(self, X, y):
        """
        X and y should be standard NumPy arrays or PyTorch Tensors.
        Converts them to torch.float32 for standard NN precision.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape y to be a column vector if it's 1D
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)

        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]

    def _calculate_r2(self, y_true, y_pred):
        """Calculates the Coefficient of Determination (R^2)."""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        # 1e-8 prevents division by zero if all target values are identical
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return r2.item()

    def _train_model_in_sample(self, model, eta, bSize, epochs=100):
        """Internal training loop to handle batching, loss, and optimization."""
        dataset = TensorDataset(self.X, self.y)
        loader = DataLoader(dataset, batch_size=bSize, shuffle=True)

        # Using Mean Squared Error for regression; use nn.BCELoss() for binary classification
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=eta)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()           # Clear old gradients
                predictions = model(batch_X)    # Forward pass
                loss = criterion(predictions, batch_y) # Compute Loss
                loss.backward()                 # Backward pass
                optimizer.step()                # Update weights

        # Calculate final in-sample loss
        model.eval()
        with torch.no_grad():
            final_preds = model(self.X)
            final_loss = criterion(final_preds, self.y).item()
            r2_score = self._calculate_r2(self.y, final_preds)

            print(f"Final In-Sample MSE Loss: {final_loss:.4f} | R^2: {r2_score:.4f}")

        return model

    def _train_model_tts(self, model, eta, bSize, epochs=100):
        """Splits data 80/20, trains on 80%, and evaluates Out-of-Sample Loss on 20%."""
        dataset = TensorDataset(self.X, self.y)

        # Handle the 80/20 Split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders for both sets
        train_loader = DataLoader(train_dataset, batch_size=bSize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bSize, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=eta)

        # Training Phase (on 80% only)
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation Phase (on 20% unseen data)
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            all_preds = []
            all_targets = []

            for batch_X, batch_y in test_loader:
                preds = model(batch_X)
                test_loss += criterion(preds, batch_y).item() * batch_X.size(0)

                # Store predictions and targets to calculate R^2 for the whole test set
                all_preds.append(preds)
                all_targets.append(batch_y)

            # Concatenate lists into single tensors
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            test_loss /= test_size
            r2_score = self._calculate_r2(all_targets, all_preds)

            print(f"Final Test (Out-of-Sample) MSE Loss: {test_loss:.4f | R^2: {r2_score:.4f}}")

        return model

    def runNeuralNet2L(self, eta=0.1, bSize=20, f=torch.sigmoid, epochs=100):
        print("\n--- Training NeuralNet_2L ---")
        model = NeuralNet_2L(self.input_dim, self.output_dim, f)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)

    def runNeuralNet3L(self, eta=0.1, bSize=20, nz=3, f1=torch.sigmoid, f2=torch.sigmoid, epochs=100):
        print("\n--- Training NeuralNet_3L ---")
        model = NeuralNet_3L(self.input_dim, nz, self.output_dim, f1, f2)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)

    def runNeuralNet4L(self, eta=0.1, bSize=20, nz1=4, nz2=3, f1=torch.sigmoid, f2=torch.sigmoid, f3=torch.sigmoid, epochs=100):
        print("\n--- Training NeuralNet_4L ---")
        model = NeuralNet_4L(self.input_dim, nz1, nz2, self.output_dim, f1, f2, f3)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)