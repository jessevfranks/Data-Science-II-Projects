import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


class NeuralNet_2L(nn.Module):
    """2 Layers: Input -> Output (Sigmoid for classification)"""
    def __init__(self, input_dim, output_dim):
        super(NeuralNet_2L, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.layer1(x))


class NeuralNet_3L(nn.Module):
    """3 Layers: Input -> Hidden -> Output"""
    def __init__(self, input_dim, hidden_dim, output_dim, f1=nn.ReLU()):
        super(NeuralNet_3L, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            f1,
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class NeuralNet_4L(nn.Module):
    """4 Layers: Input -> Hidden 1 -> Hidden 2 -> Output"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, f1=nn.ReLU(), f2=nn.ReLU()):
        super(NeuralNet_4L, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            f1,
            nn.Linear(hidden_dim1, hidden_dim2),
            f2,
            nn.Linear(hidden_dim2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class NeuralNet_Deep(nn.Module):
    """
    Advanced Architecture:
    Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x 2 -> Sigmoid Output
    """
    def __init__(self, input_dim, output_dim, nz1=16, nz2=8):
        super(NeuralNet_Deep, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, nz1),
            nn.BatchNorm1d(nz1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(nz1, nz2),
            nn.BatchNorm1d(nz2),
            nn.ReLU(),
            nn.Linear(nz2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class PyTorchClassificationUtils:
    """
    Classification-specific version of PyTorchNetUtils.
    Uses BCE loss, sigmoid output activations, and reports accuracy.
    """

    def __init__(self, X, y):
        """
        X and y should be standard NumPy arrays or PyTorch Tensors.
        y should contain binary labels (0 or 1).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)

        self.input_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]

    def _calculate_accuracy(self, y_true, y_pred):
        """Calculates binary classification accuracy."""
        preds_binary = (y_pred > 0.5).float()
        return (preds_binary == y_true).float().mean().item()

    def _train_model_in_sample(self, model, eta, bSize, epochs=100):
        """Internal training loop: trains on full data, reports in-sample metrics."""
        dataset = TensorDataset(self.X, self.y)
        loader = DataLoader(dataset, batch_size=bSize, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=eta)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Calculate final in-sample metrics
        model.eval()
        with torch.no_grad():
            final_preds = model(self.X)
            final_loss = criterion(final_preds, self.y).item()
            accuracy = self._calculate_accuracy(self.y, final_preds)

            print(f"  In-Sample     -> BCE Loss: {final_loss:.4f} | Accuracy: {accuracy:.4f}")

        return model

    def _train_model_tts(self, model, eta, bSize, epochs=100):
        """Splits data 80/20, trains on 80%, evaluates on 20%."""
        dataset = TensorDataset(self.X, self.y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=bSize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bSize, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=eta)

        # Training phase (80%)
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation phase (20%)
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []

            for batch_X, batch_y in test_loader:
                preds = model(batch_X)
                all_preds.append(preds)
                all_targets.append(batch_y)

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)

            test_loss = criterion(all_preds, all_targets).item()
            accuracy = self._calculate_accuracy(all_targets, all_preds)

            print(f"  Out-of-Sample -> BCE Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

        return model

    def runNeuralNet2L(self, eta=0.01, bSize=32, epochs=100):
        print("\n--- Training NeuralNet_2L (Classification) ---")
        model = NeuralNet_2L(self.input_dim, self.output_dim)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)

    def runNeuralNet3L(self, eta=0.01, bSize=32, nz=16, f1=nn.ReLU(), epochs=100):
        print("\n--- Training NeuralNet_3L (Classification) ---")
        model = NeuralNet_3L(self.input_dim, nz, self.output_dim, f1)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)

    def runNeuralNet4L(self, eta=0.01, bSize=32, nz1=32, nz2=16, f1=nn.ReLU(), f2=nn.ReLU(), epochs=100):
        print("\n--- Training NeuralNet_4L (Classification) ---")
        model = NeuralNet_4L(self.input_dim, nz1, nz2, self.output_dim, f1, f2)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)

    def runNeuralNetDeep(self, eta=0.001, bSize=32, nz1=32, nz2=16, epochs=100):
        print(f"\n--- Training NeuralNet_Deep (nz1={nz1}, nz2={nz2}) (Classification) ---")
        model = NeuralNet_Deep(self.input_dim, self.output_dim, nz1, nz2)
        self._train_model_in_sample(model, eta, bSize, epochs)
        self._train_model_tts(model, eta, bSize, epochs)