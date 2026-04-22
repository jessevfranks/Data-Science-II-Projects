import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

RANDOM_STATE = 42


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name, cost_matrix=None):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Fit_Time_Sec": round(fit_time, 3)
    }

    if cost_matrix is not None:
        fp_cost, fn_cost = cost_matrix
        total_cost = fp * fp_cost + fn * fn_cost
        results["Total_Cost"] = total_cost
        results["Avg_Cost_Per_Row"] = round(total_cost / len(y_test), 3)

    print("\n" + "=" * 70)
    print(f"{dataset_name} - {model_name}  (fit time: {fit_time:.2f}s)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(f"                 Pred 0    Pred 1")
    print(f"    Actual 0   {tn:7d}   {fp:7d}")
    print(f"    Actual 1   {fn:7d}   {tp:7d}")
    if cost_matrix is not None:
        print(f"    Total cost ({fp_cost}*FP + {fn_cost}*FN): {total_cost}")

    return results


def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    feature_names = []

    if len(numeric_features) > 0:
        feature_names.extend(numeric_features)

    if len(categorical_features) > 0:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder = cat_pipeline.named_steps["encoder"]
        cat_names = encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_names.tolist())

    return feature_names


def show_logistic_coefficients(trained_pipeline, numeric_features, categorical_features, top_n=15):
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    classifier = trained_pipeline.named_steps["classifier"]

    feature_names = get_feature_names_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )

    coefs = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    }).sort_values("Coefficient", ascending=False)

    print("\nTop positive logistic regression coefficients:")
    print(coef_df.head(top_n).to_string(index=False))

    print("\nTop negative logistic regression coefficients:")
    print(coef_df.tail(top_n).sort_values("Coefficient").to_string(index=False))


def show_tree_importance(trained_pipeline, numeric_features, categorical_features, top_n=15):
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    classifier = trained_pipeline.named_steps["classifier"]

    feature_names = get_feature_names_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\nTop {top_n} feature importances:")
    print(importance_df.head(top_n).to_string(index=False))


def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))

    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def build_models(preprocessor):
    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_depth=5,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                class_weight="balanced",  # sklearn 1.4+; remove if on older version
                random_state=RANDOM_STATE
            ))
        ]),

        "Neural Network": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0005,
                batch_size=256,  # larger batch = faster on Taiwan's 30k rows
                learning_rate_init=0.001,
                max_iter=500,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


def load_and_prepare_uci(file_path):
    df = pd.read_csv(file_path)

    print("\nUCI columns:")
    print(df.columns.tolist())

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    possible_targets = [
        "default.payment.next.month",
        "default payment next month",
        "default_payment_next_month",
        "default",
        "Y",
        "target"
    ]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Could not find target column in UCI dataset.\nAvailable columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Integer-coded columns that are really categorical
    possible_categorical = [
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
    ]

    for col in possible_categorical:
        if col in X.columns and col not in categorical_features:
            categorical_features.append(col)
            if col in numeric_features:
                numeric_features.remove(col)

    print("\nUCI target column:", target_col)
    print("UCI categorical features:", categorical_features)
    print("UCI numeric features:", numeric_features)

    return X, y, numeric_features, categorical_features


def load_and_prepare_german(file_path):
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    df.columns = [
        "status_checking_account",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_since",
        "installment_rate",
        "personal_status_sex",
        "other_debtors_guarantors",
        "present_residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "number_existing_credits",
        "job",
        "number_people_liable",
        "telephone",
        "foreign_worker",
        "target"
    ]

    print("\nGerman columns:")
    print(df.columns.tolist())

    # Recode: 1 (good) -> 0, 2 (bad/default) -> 1
    df["target"] = df["target"].replace({1: 0, 2: 1})

    X = df.drop(columns=["target"])
    y = df["target"]

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    print("\nGerman target column: target")
    print("German categorical features:", categorical_features)
    print("German numeric features:", numeric_features)

    return X, y, numeric_features, categorical_features


def run_experiment(dataset_name, X, y, numeric_features, categorical_features, cost_matrix=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor)

    all_results = []
    trained_models = {}

    for model_name, pipeline in models.items():
        result = evaluate_model(
            model=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name,
            dataset_name=dataset_name,
            cost_matrix=cost_matrix
        )
        all_results.append(result)
        trained_models[model_name] = pipeline

    sort_cols = ["ROC_AUC", "F1", "Accuracy"]
    results_df = pd.DataFrame(all_results).sort_values(by=sort_cols, ascending=False)

    return results_df, trained_models


def main():
    # Load the RAW UCI CSV so all preprocessing happens inside the pipeline.
    # (Avoids double-scaling that occurred when loading the pre-cleaned CSV.)
    uci_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/UCI_Credit_Card_Cleaned.csv"
    german_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/german.data"

    # --- UCI Credit Card Default ---
    X_uci, y_uci, num_uci, cat_uci = load_and_prepare_uci(uci_path)
    uci_results, uci_models = run_experiment(
        dataset_name="UCI Credit Card Default",
        X=X_uci,
        y=y_uci,
        numeric_features=num_uci,
        categorical_features=cat_uci,
        cost_matrix=None  # no published cost matrix for Taiwan dataset
    )

    # --- German Credit ---
    # Cost matrix: FP cost = 1 (rejected a good applicant, lost business)
    #              FN cost = 5 (approved a bad applicant, lost money)
    X_ger, y_ger, num_ger, cat_ger = load_and_prepare_german(german_path)
    german_results, german_models = run_experiment(
        dataset_name="German Credit",
        X=X_ger,
        y=y_ger,
        numeric_features=num_ger,
        categorical_features=cat_ger,
        cost_matrix=(1, 5)
    )

    final_results = pd.concat([uci_results, german_results], ignore_index=True)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(final_results.to_string(index=False))

    # German ranked by cost (lower is better)
    if "Total_Cost" in german_results.columns:
        print("\n" + "=" * 80)
        print("GERMAN CREDIT: RANKED BY COST (lower = better for the bank)")
        print("=" * 80)
        cost_ranked = german_results.sort_values("Total_Cost", ascending=True)
        print(cost_ranked[["Model", "Accuracy", "F1", "ROC_AUC", "FP", "FN", "Total_Cost", "Avg_Cost_Per_Row"]].to_string(index=False))

    output_path = "/Users/aidyn/Desktop/ScalaTion-Projects/data/finalProject/credit_default_model_results.csv"
    final_results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # --- Interpretability outputs ---
    print("\n" + "=" * 80)
    print("UCI: Logistic Regression Coefficients")
    print("=" * 80)
    show_logistic_coefficients(uci_models["Logistic Regression"], num_uci, cat_uci)

    print("\n" + "=" * 80)
    print("UCI: Random Forest Feature Importance")
    print("=" * 80)
    show_tree_importance(uci_models["Random Forest"], num_uci, cat_uci)

    print("\n" + "=" * 80)
    print("German: Logistic Regression Coefficients")
    print("=" * 80)
    show_logistic_coefficients(german_models["Logistic Regression"], num_ger, cat_ger)

    print("\n" + "=" * 80)
    print("German: Random Forest Feature Importance")
    print("=" * 80)
    show_tree_importance(german_models["Random Forest"], num_ger, cat_ger)


if __name__ == "__main__":
    main()