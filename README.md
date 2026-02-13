# Data Science II — Project Repository

This repository contains coursework and projects developed for **Data Science II**.
Projects focus on statistical modeling, regression analysis, and comparative evaluation of classical and modern techniques using **Scala** and **Python**.

---

## Repository Purpose

The goal of this folder is to serve as a **central collection of all Data Science II projects**, including:

* Project 1 (current): Implementation of regression-based modeling techniques
* Project 2:
* Project 3:
* Term Project:
 
---

## Languages & Software

### Languages

* **Scala** (primary analytical modeling via ScalaTion)
* **Python** (statistical modeling, validation, and experimentation)

### Software / Libraries

* **ScalaTion** — statistical and optimization library for Scala
* **statsmodels (Python)** — classical statistical modeling
* Standard Python stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

---

## Project 1 (Currently In Progress)

### Objective:

Evaluate and compare multiple regression strategies under **multicollinearity** and **feature transformation** scenarios.

### Modeling Techniques:

* Linear Regression [ 1 model ]
* Regularized Regression: Ridge (L2), Lasso (L1) [ 2 models ]
* Transformed Regression: Sqrt, Log1p, Box-Cox or Yeo-Johnson [ 3 models ]
* Symbolic Regression: (a) SymRidgeRegression, (b) PySR [ 1 model ]

### Datasets:

1. **Auto MPG Dataset** *(398 × 7)*
   https://archive.ics.uci.edu/dataset/9/auto+mpg

2. **House Price Regression Dataset** *(1000 × 8)*
   https://www.kaggle.com/datasets/prokshitha/home-value-insights

3. **Group-Selected Dataset** *(4898 × 11)*
   https://archive.ics.uci.edu/dataset/186/wine+quality

---

## Analytical Workflow:

Each project follows a structured data science pipeline:

**1. Data Preprocessing**

* Missing value handling
* Outlier detection and treatment
* Feature scaling / normalization (when required)

**2. Exploratory Data Analysis (EDA)**

* Distribution analysis
* Correlation structure
* Multicollinearity diagnostics (VIF, condition number)
* Visualization of relationships

**3. Model Training & Evaluation**

* In-Sample Fit
* Validation Split
* Cross-Validation (linear regression)

**4. Feature Selection**

* Forward Selection
* Backward Elimination
* Stepwise Selection

**5. Statistical Summaries & Visualization**

* Coefficient interpretation
* Residual plots
* Regularization paths
* Model comparison charts

---

## Learning Objectives

This repository emphasizes:

* Practical implementation of statistical theory
* Cross-platform reproducibility
* Understanding regression beyond OLS
* Diagnosing and correcting model violations
* Interpretable machine learning through symbolic regression

---
