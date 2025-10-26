# 📈 Multiple Linear Regression using Stochastic Gradient Descent (SGD)

This project implements a **Multiple Linear Regression** model **from scratch** using **Stochastic Gradient Descent (SGD)** — without using any machine learning libraries such as scikit-learn or TensorFlow.

---

## 🧠 Project Overview

The goal of this project is to understand how Linear Regression works internally by manually implementing all its core steps — including predictions, error calculation, and parameter updates.

The dataset used (`MultipleLR.csv`) contains **three input features** and **one target output**.  
It represents a simple multiple linear relationship as shown below:

| X1 | X2 | X3 | Y |
|----|----|----|---|
| 73 | 80 | 75 | 152 |
| 93 | 88 | 93 | 185 |
| 89 | 91 | 90 | 180 |
| 96 | 98 | 100 | 196 |
| ... | ... | ... | ... |

---

## ⚙️ Code Structure and Explanation

### 1️⃣ Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
### 2️⃣ Loading and Preparing the Data
```python
data = pd.read_csv("MultipleLR.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
```
### 3️⃣ Initializing Parameters
```python
m, n = X.shape
weights = np.zeros(n)
bias = 0
alpha = 0.0001
epochs = 100
```
### 4️⃣ Training using Stochastic Gradient Descent (SGD)
```python
for epoch in range(epochs):
    for i in range(m):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights) + bias
        error = yi - y_pred
        weights += alpha * error * xi
        bias += alpha * error
```
### 5️⃣ Displaying Model Parameters
```python
print("Final weights:", weights)
print("Final bias:", bias)
```
### 6️⃣ Making Predictions and Evaluating Results
```python
y_pred_all = np.dot(X, weights) + bias
comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": np.round(y_pred_all, 2)
})
print(comparison.head(10))
```

### 7️⃣ Visualizing the Performance
```python
plt.figure(figsize=(8,5))
plt.plot(y, label="Actual", marker='o')
plt.plot(y_pred_all, label="Predicted", marker='x')
plt.title("Multiple Linear Regression using SGD")
plt.xlabel("Samples")
plt.ylabel("Y values")
plt.legend()
plt.grid(True)
plt.show()
```

### 📊 Example Output
```sql
Final weights: [0.58088278 0.5854511  0.85592962]
Final bias: 0.007537626110254033

Comparison between actual and predicted:
   Actual  Predicted
0     152     153.44
1     185     185.15
2     180     182.02
3     196     198.74
4     142     140.97
5     101     104.80
6     149     149.32
7     115     111.45
8     175     173.83
9     164     162.20
```
<img width="521" height="347" alt="image" src="https://github.com/user-attachments/assets/7dc20cd8-233a-4ab6-87fa-db877c921995" />

# 10 Machine Learning Optimizers (one-line explanations)

1. **Stochastic Gradient Descent (SGD)** — Updates parameters using the gradient of a single (or small) sample each step; simple and memory-efficient but can be slow and noisy.

2. **Mini-batch SGD** — A compromise between SGD and batch gradient descent that updates using small batches for stability and speed.

3. **Momentum** — Accelerates convergence by accumulating an exponentially decayed sum of past gradients to smooth updates.

4. **Nesterov Accelerated Gradient (NAG)** — A variant of momentum that computes the gradient after a look-ahead step for improved convergence.

5. **Adagrad** — Adapts learning rates per-parameter based on historical squared gradients; good for sparse data but learning rate decays heavily.

6. **RMSprop** — Improves Adagrad by using an exponential moving average of squared gradients to prevent aggressive decay.

7. **Adadelta** — Extension of Adagrad that restricts the window of accumulated past gradients and eliminates a default learning rate.

8. **Adam (Adaptive Moment Estimation)** — Combines momentum and RMSprop ideas: adaptive per-parameter learning rates with momentum-like first and second moment estimates.

9. **AdamW** — Adam variant that decouples weight decay from the gradient-based update for better regularization behavior.

10. **Nadam** — Adam optimizer enhanced with Nesterov momentum for potentially faster and more stable convergence.

---

*Short note:* These optimizers differ mainly in how they scale step sizes and use past gradients; choice depends on problem, data sparsity, and model architecture.

