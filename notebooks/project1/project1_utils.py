import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures

class Project1Utils:
    def __init__(self, X, y):
        """
        X: pandas DataFrame or numpy array (Predictors)
        y: pandas Series or numpy array (Target)
        feature_names: list of strings (optional, inferred from df if possible)
        """
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()

    # --- Helper: Add Constant ---
    # Statsmodels requires manually adding the intercept (column of 1s)
    def _get_X_with_const(self, X_data):
        return sm.add_constant(X_data)

    def run_linear_regressions(self):
        print("Running Linear Regression")

        print("In-Sample")
        X_const = self._get_X_with_const(self.X)
        model = sm.OLS(self.y, X_const).fit()
        print(model.summary())

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train_c = self._get_X_with_const(X_train)
        X_test_c = self._get_X_with_const(X_test)

        model_tt = sm.OLS(y_train, X_train_c).fit()
        y_pred = model_tt.predict(X_test_c)

        # Manually calc R2 for test set
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_test = 1 - (ss_res / ss_tot)
        print(f"Test Set R-squared: {r2_test:.4f}")

        # 5-Fold Cross Validation
        print("5-Fold Cross Validation")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []

        for train_idx, val_idx in kf.split(self.X):
            X_fold_train, X_fold_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_fold_train, y_fold_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model_cv = sm.OLS(y_fold_train, self._get_X_with_const(X_fold_train)).fit()
            pred_cv = model_cv.predict(self._get_X_with_const(X_fold_val))

            # Calc R2
            ss_res_cv = np.sum((y_fold_val - pred_cv) ** 2)
            ss_tot_cv = np.sum((y_fold_val - np.mean(y_fold_val)) ** 2)
            r2_scores.append(1 - (ss_res_cv / ss_tot_cv))

        print(f"Average CV R-squared: {np.mean(r2_scores):.4f}")

    def run_ridge_regressions(self, alpha=1.0):
        print(f"Running Ridge Regression")

        # Prepare Data with Constant
        X_const = self._get_X_with_const(self.X)

        # Create Alpha Vector to exempt Intercept
        # We pass an array of alphas: 0 for the const (index 0), 'alpha' for the rest.
        k = X_const.shape[1]
        alpha_vec = np.zeros(k)
        alpha_vec[1:] = alpha

        model = sm.OLS(self.y, X_const).fit_regularized(method='elastic_net', alpha=alpha_vec, L1_wt=0.0)

        fitted_values = model.predict(X_const)
        ss_res = np.sum((self.y - fitted_values) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"R-squared: {r_squared:.4f}")

    def run_lasso_regressions(self, alpha=1.0):
        print(f"Run Lasso Regression")

        # Prepare Data with Constant
        X_const = self._get_X_with_const(self.X)

        # Create Alpha Vector to exempt Intercept
        k = X_const.shape[1]
        alpha_vec = np.zeros(k)
        alpha_vec[1:] = alpha

        model = sm.OLS(self.y, X_const).fit_regularized(method='elastic_net', alpha=alpha_vec, L1_wt=1.0)

        fitted_values = model.predict(X_const)
        ss_res = np.sum((self.y - fitted_values) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"R-squared: {r_squared:.4f}")

    def run_transformed_regressions(self):
        print("Running Transformed Regressions")
        X_const = self._get_X_with_const(self.X)

        print("Sqrt(y)")
        y_sqrt = np.sqrt(self.y)
        model_sqrt = sm.OLS(y_sqrt, X_const).fit()
        print(f"R-squared: {model_sqrt.rsquared:.4f}")

        print("Log1p(y)")
        y_log = np.log1p(self.y)
        model_log = sm.OLS(y_log, X_const).fit()
        print(f"R-squared: {model_log.rsquared:.4f}")

        print("Box-Cox(y)")
        # boxcox requires positive data. It returns (transformed_data, best_lambda)
        y_bc, fitted_lambda = stats.boxcox(self.y[self.y > 0])
        model_bc = sm.OLS(y_bc, X_const.loc[self.y > 0]).fit()
        print(f"Best Lambda: {fitted_lambda:.4f}")
        print(f"R-squared: {model_bc.rsquared:.4f}")

        print("Yeo-Johnson(y)")
        y_yj, fitted_lambda_yj = stats.yeojohnson(self.y)
        model_yj = sm.OLS(y_yj, X_const).fit()
        print(f"Best Lambda: {fitted_lambda_yj:.4f}")
        print(f"R-squared: {model_yj.rsquared:.4f}")


    def run_sym_ridge_regression(self):
        print("SymRidge (Quadratic) Regression")
        # 1. Expand Features (x1, x2 -> 1, x1, x2, x1^2, x1x2, x2^2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.X)

        model = sm.OLS(self.y, X_poly).fit_regularized(method='elastic_net', alpha=1.0, L1_wt=0.0)
        fitted_values = model.predict(X_poly)
        ss_res = np.sum((self.y - fitted_values) ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"R-squared: {r_squared:.4f}")