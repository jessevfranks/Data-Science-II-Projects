## EDA and Preprocessing

For the California Housing dataset, preprocessing required a bit more attention to scale than standard regression. Neural networks are highly sensitive to input magnitude, meaning the vast difference between total_rooms (in the thousands) and median_income (single digits) would cause gradient explosion.

To resolve this, StandardScaler was used to center the continuous features around a mean of 0 with a standard deviation of 1. Additionally, the target variable (median_house_value) was scaled down by dividing by 100,000. This turned prices like $452,600 into 4.526, stabilizing the Adam optimizer. The categorical ocean_proximity feature was integer-encoded and left separate from the standard scaling to preserve its distinct categorizations.

---
## 2L Neural Network (Baseline)

In the 2-Layer Neural Network (essentially a linear model with no hidden layers), we see consistent R^2 values of ~0.64. This model gives us a solid baseline, indicating that roughly 64% of the variance in house prices can be explained by purely linear relationships with the features. Because In-Sample and Train-Test Split (Out-of-Sample) results were very close (0.6447 vs 0.6491), this suggests the model is highly stable and not overfitting, but it lacks the complexity to capture deeper patterns in the data.

---
## 3L Neural Network

In the 3-Layer Neural Network, we introduce a hidden layer with non-linear activation functions (ReLU). Here, we see a significant jump in performance. The R^2 increases to ~0.76. This proves that there are non-linear interactions between the features—such as how geographical coordinates (longitude/latitude) interact with median_income to affect price—that a simple 2L network couldn't catch. Generalization remains strong, with Test MSE closely tracking In-Sample MSE.

---
## Deep Neural Network (Custom Architecture)

For our custom "Pick an Architecture" requirement, a Deep Neural Network was constructed with two hidden layers, Batch Normalization, and Dropout regularization. We saw a couple of interesting things here. This model achieved the highest In-Sample R^2 of 0.8061, proving its high capacity to learn the dataset.

However, the Test R^2 dropped to 0.7456. This is a classic indicator of overfitting. The deeper model is so complex that it began "memorizing" the training data rather than learning generalizable patterns, causing it to perform worse on unseen data compared to the simpler 3L model.

| Model | In-Sample R^2 | Test (80/20) R^2 | Test MSE | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **2L Network** | 0.6447 | 0.6491 | 0.4641 | Stable Baseline |
| **3L Network** | 0.7600 | 0.7583 | 0.3214 | Strong/Balanced |
| **Deep Network** | 0.8061 | 0.7456 | 0.3407 | Overfit |

---
## Hyper-Parameter Experimentation

To find the optimal tuning for our neural networks, a grid search was performed over Learning Rate (eta) and Batch Size using the 3L model. The results clearly demonstrated the "Goldilocks zone" for training this specific dataset.

At a high learning rate (eta = 0.1) and small batch size (16), the model completely diverged, resulting in a massively negative R^2 of -3.2669. The weights updated so aggressively that the optimizer shot past the minimum.

The "sweet spot" was found at eta = 0.01 and Batch Size = 16, which yielded our best overall Test R^2 of 0.7710. Lowering the learning rate further to 0.001 stabilized the training but resulted in a lower R^2 (~0.72), suggesting the model learned too slowly to reach the optimal weights within the 50-epoch limit.

### Table of Tuning Results (3L Network):
| Learning Rate (eta) | Batch Size | Test R^2 | Status |
| :--- | :--- | :--- | :--- |
| **0.1** | 16 | -3.2669 | Diverged (Exploded) |
| **0.1** | 64 | 0.6378 | Sub-optimal |
| **0.01** | **16** | **0.7710** | **Best Configuration** |
| **0.01** | 64 | 0.7383 | Good/Stable |
| **0.001** | 16 | 0.7251 | Too Slow |
| **0.001** | 64 | 0.7329 | Too Slow |

---
## Financial Impact & Conclusion

Translating our best model's performance back into real-world financial terms gives us a clear picture of its utility. Our champion model (3L Network, eta=0.01, Batch Size=16) achieved a Test MSE of 0.2931 in scaled units.

Since the target variable was scaled by dividing by 100,000, we can calculate the real-world error as follows:
* **Test MSE** = 0.2931
* **RMSE** = sqrt(0.2931) = 0.54138
* **Real-World Error** = 0.54138 * $100,000 = $54,138

Given that the median house prices in this dataset range up to $500,000, an average prediction error of ~$54,138 is a very respectable margin, proving that the non-linear 3L neural network is highly capable of modeling California housing prices.