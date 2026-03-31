# Diabetes Health Indicators

### diabetes_NNet.py output:

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


### diabetesMain.py output: 

--- Training NeuralNet_2L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.4990 | R^2: -0.9962

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.5004 | R^2: -1.0016

--- Training NeuralNet_4L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.4979 | R^2: -0.9917

=== HYPERPARAMETER EXPERIMENTATION ===

>>> Testing 3L: LR=0.1, Batch Size=16

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.5004 | R^2: -1.0016

>>> Testing 3L: LR=0.1, Batch Size=64

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.4953 | R^2: -0.9814

>>> Testing 3L: LR=0.01, Batch Size=16

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.1831 | R^2: 0.2675
Final Test (Out-of-Sample) MSE Loss: 0.1761 | R^2: 0.2957

>>> Testing 3L: LR=0.01, Batch Size=64

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.1705 | R^2: 0.3179
Final Test (Out-of-Sample) MSE Loss: 0.1702 | R^2: 0.3192

>>> Testing 3L: LR=0.001, Batch Size=16

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.1673 | R^2: 0.3307
Final Test (Out-of-Sample) MSE Loss: 0.1689 | R^2: 0.3242

>>> Testing 3L: LR=0.001, Batch Size=64

--- Training NeuralNet_3L ---
Final In-Sample MSE Loss: 0.5000 | R^2: -1.0000
Final Test (Out-of-Sample) MSE Loss: 0.4994 | R^2: -0.9976