"""
Standalone PyTorch neural network benchmark with hyperparameter tuning.

Two phases:
  1. Architecture comparison: no-hidden / 1-hidden / 2-hidden, default hyperparameters.
  2. Hyperparameter tuning: on the best architecture per dataset, grid-search
     learning rate and dropout using 3-fold cross-validation.

Phase 1 answers "does depth help?" Phase 2 answers "given the best depth,
how much can tuning squeeze out?" — both relevant to the report.
"""

import pandas as pd
import numpy as np
import time
from itertools import product

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from credit_default_models import (
    load_and_prepare_uci,
    load_and_prepare_german,
    build_preprocessor,
    evaluate_model,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# PyTorch MLP wrapped in a sklearn-compatible interface.
# ---------------------------------------------------------------------------
class PyTorchMLP(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible PyTorch neural network.

    hidden_sizes controls depth:
      ()         -> no hidden layers (linear model, like logistic regression)
      (64,)      -> one hidden layer with 64 units
      (64, 32)   -> two hidden layers with 64 and 32 units
    """

    def __init__(self, hidden_sizes=(64, 32), dropout=0.2, lr=0.001,
                 epochs=100, batch_size=256, class_weight=None,
                 device=None, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state

    def _build_model(self, input_dim):
        if len(self.hidden_sizes) == 0:
            return nn.Linear(input_dim, 2)

        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        X_tensor = torch.from_numpy(X).to(device)
        y_tensor = torch.from_numpy(y).to(device)

        self.model_ = self._build_model(X.shape[1]).to(device)

        if self.class_weight == "balanced":
            counts = np.bincount(y)
            weights = len(y) / (len(counts) * counts)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        n = len(X)
        self.model_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = np.ascontiguousarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
ARCHITECTURES = [
    ((),            "No Hidden Layers"),
    ((64,),         "1 Hidden Layer (64)"),
    ((64, 32),      "2 Hidden Layers (64->32)"),
]

# Hyperparameter grid for Phase 2. Kept small to stay reasonable in runtime.
LEARNING_RATES = [0.0005, 0.001, 0.005]
DROPOUTS = [0.1, 0.3, 0.5]


def run_single_training(X_train, X_test, y_train, y_test,
                         numeric_features, categorical_features,
                         hidden_sizes, label, dataset_name,
                         cost_matrix=None, lr=0.001, dropout=0.2, epochs=100):
    """Train one configuration end-to-end and return its result dict."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", PyTorchMLP(
            hidden_sizes=hidden_sizes,
            dropout=dropout if len(hidden_sizes) > 0 else 0.0,
            lr=lr,
            epochs=epochs,
            batch_size=256,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ))
    ])

    return evaluate_model(
        model=pipeline,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        model_name=label,
        dataset_name=dataset_name,
        cost_matrix=cost_matrix,
    )


def cross_validate_config(X_train, y_train, numeric_features, categorical_features,
                           hidden_sizes, lr, dropout, n_folds=3, epochs=100):
    """Run stratified k-fold CV on the training set and return mean F1 on minority class."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    # Ensure we work with the underlying arrays so iloc works on DataFrames/Series
    X_arr = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else X_train
    y_arr = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else y_train

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr = X_arr.iloc[train_idx] if hasattr(X_arr, "iloc") else X_arr[train_idx]
        X_vl = X_arr.iloc[val_idx] if hasattr(X_arr, "iloc") else X_arr[val_idx]
        y_tr = y_arr.iloc[train_idx] if hasattr(y_arr, "iloc") else y_arr[train_idx]
        y_vl = y_arr.iloc[val_idx] if hasattr(y_arr, "iloc") else y_arr[val_idx]

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", PyTorchMLP(
                hidden_sizes=hidden_sizes,
                dropout=dropout if len(hidden_sizes) > 0 else 0.0,
                lr=lr,
                epochs=epochs,
                batch_size=256,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ))
        ])
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_vl)
        fold_scores.append(f1_score(y_vl, y_pred, zero_division=0))

    return np.mean(fold_scores), np.std(fold_scores)


def tune_hyperparameters(X_train, y_train, numeric_features, categorical_features,
                          hidden_sizes, dataset_name, epochs=100):
    """Grid-search lr x dropout on the given architecture. Returns best (lr, dropout)."""
    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING — {dataset_name} — architecture {hidden_sizes}")
    print("=" * 70)
    print(f"Grid: lr in {LEARNING_RATES}, dropout in {DROPOUTS}")
    print(f"CV folds: 3, scoring: F1 on minority class\n")

    results = []
    best_score = -np.inf
    best_config = None

    for lr, dropout in product(LEARNING_RATES, DROPOUTS):
        t0 = time.perf_counter()
        mean_f1, std_f1 = cross_validate_config(
            X_train, y_train, numeric_features, categorical_features,
            hidden_sizes=hidden_sizes, lr=lr, dropout=dropout, n_folds=3, epochs=epochs
        )
        elapsed = time.perf_counter() - t0
        results.append({
            "lr": lr, "dropout": dropout,
            "mean_f1": round(mean_f1, 4), "std_f1": round(std_f1, 4),
            "cv_time_sec": round(elapsed, 2),
        })
        print(f"  lr={lr:<8}  dropout={dropout:<5}  F1 = {mean_f1:.4f} +/- {std_f1:.4f}  ({elapsed:.1f}s)")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_config = (lr, dropout)

    print(f"\n  Best config: lr={best_config[0]}, dropout={best_config[1]}  (CV F1 = {best_score:.4f})")
    return best_config, pd.DataFrame(results)


def run_experiments_for_dataset(dataset_name, X, y, numeric_features,
                                 categorical_features, cost_matrix=None, epochs=100):
    """Full pipeline for one dataset: architecture sweep, then tuning of the winner."""
    print("\n" + "#" * 80)
    print(f"# {dataset_name}")
    print("#" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Phase 1: architecture sweep with default hyperparameters 
    print("\n" + "=" * 70)
    print(f"PHASE 1: ARCHITECTURE COMPARISON ({dataset_name})")
    print("=" * 70)

    phase1_results = []
    for arch, label in ARCHITECTURES:
        result = run_single_training(
            X_train, X_test, y_train, y_test,
            numeric_features, categorical_features,
            hidden_sizes=arch,
            label=f"PyTorch NN — {label} (untuned)",
            dataset_name=dataset_name,
            cost_matrix=cost_matrix,
            lr=0.001, dropout=0.2, epochs=epochs,
        )
        result["hidden_sizes"] = str(arch)
        result["phase"] = "1-untuned"
        phase1_results.append(result)

    phase1_df = pd.DataFrame(phase1_results)
    best_idx = phase1_df["F1"].idxmax()
    best_arch_str = phase1_df.loc[best_idx, "hidden_sizes"]
    # Map string back to tuple
    best_arch = eval(best_arch_str)
    best_arch_label = phase1_df.loc[best_idx, "Model"]
    print(f"\nBest architecture on {dataset_name}: {best_arch_label} (F1 = {phase1_df.loc[best_idx, 'F1']:.4f})")

    # --- Phase 2: hyperparameter tuning on the best architecture ---
    # Skip tuning for no-hidden (no dropout to tune, and linear models have minimal lr sensitivity)
    if len(best_arch) == 0:
        print("\nBest architecture has no hidden layers; skipping dropout tuning.")
        print("Running a small LR sweep only.\n")
        tuning_grid = pd.DataFrame({"lr": LEARNING_RATES})
        best_lr = 0.001
        best_score = -np.inf
        for lr in LEARNING_RATES:
            mean_f1, std_f1 = cross_validate_config(
                X_train, y_train, numeric_features, categorical_features,
                hidden_sizes=best_arch, lr=lr, dropout=0.0, n_folds=3, epochs=epochs
            )
            print(f"  lr={lr:<8}  F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
            if mean_f1 > best_score:
                best_score = mean_f1
                best_lr = lr
        best_dropout = 0.0
        print(f"\n  Best lr: {best_lr} (CV F1 = {best_score:.4f})")
    else:
        (best_lr, best_dropout), tuning_grid = tune_hyperparameters(
            X_train, y_train, numeric_features, categorical_features,
            hidden_sizes=best_arch,
            dataset_name=dataset_name,
            epochs=epochs,
        )

    # Part 3: re-fit tuned model on full training set, evaluate on test set
    print("\n" + "=" * 70)
    print(f"PHASE 2: TUNED MODEL — FINAL TEST-SET EVALUATION ({dataset_name})")
    print("=" * 70)

    clean_arch_label = best_arch_label.replace("PyTorch NN —", "").replace("(untuned)", "").strip()

    tuned_result = run_single_training(
        X_train, X_test, y_train, y_test,
        numeric_features, categorical_features,
        hidden_sizes=best_arch,
        label=f"PyTorch NN — {clean_arch_label} (TUNED lr={best_lr}, dropout={best_dropout})",
        dataset_name=dataset_name,
        cost_matrix=cost_matrix,
        lr=best_lr, dropout=best_dropout, epochs=epochs,
    )
    tuned_result["hidden_sizes"] = str(best_arch)
    tuned_result["phase"] = "2-tuned"

    return phase1_df, tuning_grid, tuned_result



def main():
    if torch.cuda.is_available():
        device_note = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_note = "Apple Silicon GPU (MPS)"
    else:
        device_note = "CPU"
    print(f"PyTorch device: {device_note}")

    uci_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/UCI_Credit_Card_Cleaned.csv"
    german_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/german.data"

    # UCI
    X_uci, y_uci, num_uci, cat_uci = load_and_prepare_uci(uci_path)
    uci_phase1, uci_tuning, uci_tuned = run_experiments_for_dataset(
        "UCI Credit Card Default", X_uci, y_uci, num_uci, cat_uci,
        cost_matrix=None, epochs=100,
    )

    # German 
    X_ger, y_ger, num_ger, cat_ger = load_and_prepare_german(german_path)
    ger_phase1, ger_tuning, ger_tuned = run_experiments_for_dataset(
        "German Credit", X_ger, y_ger, num_ger, cat_ger,
        cost_matrix=(1, 5), epochs=100,
    )

    # summary
    print("\n\n" + "#" * 80)
    print("# CONSOLIDATED SUMMARY")
    print("#" * 80)

    all_results = pd.concat([uci_phase1, ger_phase1], ignore_index=True)
    tuned_df = pd.DataFrame([uci_tuned, ger_tuned])
    all_results = pd.concat([all_results, tuned_df], ignore_index=True)

    print("\nALL PHASE 1 + PHASE 2 RESULTS (test-set metrics):")
    cols_to_show = ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1",
                     "TN", "FP", "FN", "TP", "Fit_Time_Sec"]
    if "Total_Cost" in all_results.columns:
        cols_to_show.append("Total_Cost")
    print(all_results[cols_to_show].to_string(index=False))

    print("\nTUNING GRID — UCI:")
    print(uci_tuning.to_string(index=False))
    print("\nTUNING GRID — German:")
    print(ger_tuning.to_string(index=False))

    print("\nBEST MODEL PER DATASET (by test-set F1):")
    best = all_results.loc[all_results.groupby("Dataset")["F1"].idxmax()]
    print(best[cols_to_show].to_string(index=False))

    output_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/pytorch_model_results.csv"
    all_results.to_csv(output_path, index=False)
    print(f"\nSaved all results to: {output_path}")


if __name__ == "__main__":
    main()