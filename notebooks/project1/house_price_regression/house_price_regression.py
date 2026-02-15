import pandas as pd
import os

from project1.project1_utils import Project1Utils


def main():
    file_path = os.path.expanduser('../../../data/project1/house_price_regression_dataset.csv')

    df = pd.read_csv(file_path)

    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X = df[feature_cols]
    y = df[target_col]

    # Run
    utils = Project1Utils(X, y)
    utils.run_linear_regressions()
    utils.run_ridge_regressions(alpha=0.5)
    utils.run_lasso_regressions(alpha=0.5)
    utils.run_transformed_regressions()
    utils.run_sym_ridge_regression()

if __name__ == "__main__":
    main()