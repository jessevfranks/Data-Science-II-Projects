# Diabetes Health Indicators

I created two python files, diabetes_NNet (standalone PyTorch code) and diabetesMain (PyTorchNetUtils), so that I could compare the accuracy and outputs. 

---
## diabetes_NNet.py output:

Model Comparison:
Model | Train Acc | Test Acc
-----------------------------
2L    | 0.7470     | 0.7465
3L    | 0.7533     | 0.7506
4L    | 0.7591     | 0.7521

Learning Rate Tuning (3L Model):
LR    | Train Acc | Test Acc
-----------------------------
0.1   | 0.7278     | 0.7258
0.01  | 0.7529     | 0.7517
0.001 | 0.7569     | 0.7539


### Model Comparison: 
All three architectures learned meaningfully. The 2L model (basically logistic regression) got a ~74.7% accuracy, which tells me that there's a pretty solid linear signal in the data. Adding hidden layers helps incrementally, 3L gets ~75.1% and 4L gets ~75.2% on test. The small gap between train and test accuracy across all models means there's no overfitting, which is good.

### Learning Rate Tuning: 
0.001 beat 0.01 and 0.1 on both train and test accuracy. The 0.1 rate hurt performance (74.7% -> 72.6%), which suggests it was overshooting the loss landscape and bouncing around the minimum instead of settling into it. The slower 0.001 rate let the optimizer converge more precisely.

### Overall takeaway: 
~75% accuracy on a balanced binary dataset with 21 features is a reasonable result. It's beating random guessing (50%) by a solid margin, but the ceiling around 75% suggests that diabetes prediction from survey data has inherent noise, people with similar risk factor profiles can have different outcomes.

---
## diabetesMain.py output: 

--- Training NeuralNet_2L (Classification) ---
  In-Sample     -> BCE Loss: 0.5162 | Accuracy: 0.7458
  Out-of-Sample -> BCE Loss: 0.5180 | Accuracy: 0.7426

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5257 | Accuracy: 0.7432
  Out-of-Sample -> BCE Loss: 0.5094 | Accuracy: 0.7481

--- Training NeuralNet_4L (Classification) ---
  In-Sample     -> BCE Loss: 0.5032 | Accuracy: 0.7529
  Out-of-Sample -> BCE Loss: 0.5024 | Accuracy: 0.7530

=== HYPERPARAMETER EXPERIMENTATION ===

>>> Testing 3L: LR=0.1, Batch Size=16

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.6944 | Accuracy: 0.5000
  Out-of-Sample -> BCE Loss: 0.6931 | Accuracy: 0.5018

>>> Testing 3L: LR=0.1, Batch Size=64

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5191 | Accuracy: 0.7449
  Out-of-Sample -> BCE Loss: 0.5312 | Accuracy: 0.7406

>>> Testing 3L: LR=0.01, Batch Size=16

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5144 | Accuracy: 0.7479
  Out-of-Sample -> BCE Loss: 0.5110 | Accuracy: 0.7474

>>> Testing 3L: LR=0.01, Batch Size=64

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5112 | Accuracy: 0.7486
  Out-of-Sample -> BCE Loss: 0.5078 | Accuracy: 0.7506

>>> Testing 3L: LR=0.001, Batch Size=16

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5004 | Accuracy: 0.7546
  Out-of-Sample -> BCE Loss: 0.5169 | Accuracy: 0.7404

>>> Testing 3L: LR=0.001, Batch Size=64

--- Training NeuralNet_3L (Classification) ---
  In-Sample     -> BCE Loss: 0.5032 | Accuracy: 0.7518
  Out-of-Sample -> BCE Loss: 0.5060 | Accuracy: 0.7521

### Diabetes Health Indicators: Analysis
The dataset contains 70,692 balanced (50/50) survey responses from the CDC's BRFSS2015 with 21 feature variables predicting binary diabetes status. A classification-specific PyTorch pipeline was built using BCE loss and sigmoid output activation, since the shared regression utils (MSE loss, ReLU output) failed to converge on binary targets — producing 50% accuracy (random guessing) due to mismatched loss signals.

### Model comparison:
The progression is 2L (74.3%) -> 3L (74.8%) -> 4L (75.3%). Deeper architectures provided marginal gains, suggesting the dataset has some nonlinear structure but not enough to dramatically benefit from added depth. The narrow train-test gap across all models indicates no overfitting.

### Hyperparameter tuning highlights:
LR=0.1 caused divergence (50% accuracy with small batches). LR=0.001 with batch size 64 achieved the best balance of accuracy and generalization. LR=0.001 with batch size 16 showed slight overfitting (75.5% train vs 74.0% test). The ~75% accuracy ceiling likely reflects inherent noise in self-reported survey data — individuals with similar risk profiles can have different diabetes outcomes.

LR=0.1 with batch size 16 completely failed (50% accuracy = random guessing on a balanced dataset). 
LR=0.1 with batch size 64 recovered a tiny bit (74.1%), which tells me that 0.1 is too aggressive.
LR=0.01 is stable across both batch sizes (~74.8–75.1%).
LR=0.001 achieved the best in-sample accuracy (75.5%) but test accuracy dropped to 74.0% with batch size 16, that's a slight overfitting signal. With batch size 64 it balanced out better (75.2% test).
Best config: LR=0.001, batch size 64. It has good accuracy on both in-sample and out-of-sample with a small overfitting gap.

---
### Classification vs. Regression:
(what I had to change in the code)
AutoMPG and California Housing are both regression problems, they're predicting continous numbers. This diabetes health indicators dataset is a classification problem, so it has some different requirements. 
The original PyTorchNetUtils class in project2_utils.py had MSE, ReLU outputs, and R² hardcoded for our regression problems. Using this class returned poor R² values for this diabetes dataset, so I created the PyTorchClassificationUtils class in project2_classification_utils.py file to have a classification-specific loss function, output activiation, and evaluation metric.

Loss Function:
Regression uses MSE where classification uses BCE. 
BCE heavily penalizes confident, wrong predictions and it provides steeper gradients resulting in faster convergence in classification. When you use MSE on binary 0/1 targets (classification problems), the gradients are weak and the model struggles to learn, so I was getting R²=-1.0 and MSE=0.5 for this dataset. Using MSE for classification had a poor performance since it doesn't penalize incorrect binary predictions strongly enough.

Output Activation:
Regression can output any number, so ReLU on the output is fine. Classification needs probabilities between 0 and 1, so the output layer needs sigmoid. Without it, the model can output values like 3.7 or -2.1, which makes no sense for "probability of diabetes."

Evaluation Metric:
R² measures how well you explain variance in a continuous target (used for evaluating regression model accuracy). It makes sense for house prices but not for binary labels. Accuracy (what percentage was classified correctly) is the better metric for classification.