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

## Project 1

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

## Learning Objectives

This repository emphasizes:

* Practical implementation of statistical theory
* Cross-platform reproducibility
* Understanding regression beyond OLS
* Diagnosing and correcting model violations
* Interpretable machine learning through symbolic regression

---

## Project 2
### Objective:
Evaluate and compare multiple **neural network architectures** under varying depth, activation functions, and hyper-parameter configurations.

### Modeling Techniques:
* 2L Neural Network — zero hidden layers *(ScalaTion + PyTorch)* [ 1 model ]
* 3L Neural Network — one hidden layer *(ScalaTion + PyTorch)* [ 1 model ]
* 4L Neural Network — two hidden layers *(PyTorch only)* [ 1 model ]
* Custom Architecture — group-selected design *(PyTorch only)* [ 1 model ]

### Hyper-Parameter Tuning:
* Number of nodes in hidden layer(s)
* Activation functions (sigmoid, tanh, ReLU, identity, etc.)
* Mini-batch size
* Learning rate (eta)

### Datasets:
1. **Auto MPG Dataset** *(398 × 7)*
   https://archive.ics.uci.edu/dataset/9/auto+mpg
2. **California Housing Prices Dataset** *(20,640 × 10)*
   https://www.kaggle.com/datasets/camnugent/california-housing-prices
3. **Group-Selected Dataset**

## Analytical Workflow:
Each project follows a structured data science pipeline:

**1. Data Preprocessing**
* Missing value handling
* Outlier detection and treatment
* Feature scaling / normalization (required for neural networks)

**2. Exploratory Data Analysis (EDA)**
* Distribution analysis
* Correlation structure
* Visualization of input-output relationships

**3. Model Training & Evaluation**
* In-Sample Fit
* Validation Split (80/20 train-test)
* Auto-tuning of learning rate via `trainNtest2`

**4. Feature Selection (Bonus)**
* Forward Selection
* Backward Elimination
* Stepwise Selection

**5. Statistical Summaries & Visualization**
* Quality of Fit metrics (R², SMAPE, MAE)
* Loss curves per epoch
* Predicted vs. actual plots
* Architecture and hyper-parameter comparison charts

## Learning Objectives
This project emphasizes:
* Understanding feedforward neural networks from first principles
* Gradient descent and backpropagation mechanics
* Cross-platform implementation (ScalaTion JVM vs. PyTorch Python)
* Hyper-parameter sensitivity and tuning strategies
* Bridging classical regression and modern deep learning
