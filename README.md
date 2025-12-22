# 📊 Multiple Linear Regression from Scratch (Batch Gradient Descent)

This project implements **Multiple Linear Regression from scratch** using **Batch Gradient Descent**, without relying on machine learning libraries such as **scikit-learn** or **TensorFlow**.

The goal is to deeply understand how linear regression works internally, including training, optimization, and evaluation.

---

## 🧠 Project Overview

- ✅ Implements **Multiple Linear Regression**
- ✅ Uses **Batch Gradient Descent** optimization
- ✅ Applies **Feature Scaling (Standardization)**
- ✅ Evaluates performance using **R-squared (R²)**
- ✅ Visualizes results using **Perfect Fit (y = x) plot**
- ✅ Tracks **Loss vs Epochs**

---

## 📁 Dataset Description

The dataset (`MultipleLR.csv`) contains:

- **3 input features**: `X1`, `X2`, `X3`
- **1 target variable**: `Y`

### Example Data:

| X1 | X2 | X3 | Y   |
|----|----|----|-----|
| 73 | 80 | 75 | 152 |
| 93 | 88 | 93 | 185 |
| 89 | 91 | 90 | 180 |
| ... | ... | ... | ... |

---

## ⚙️ Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization

---

## 🧩 Implementation Details

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### 2. Load and Prepare Data

```python
data = pd.read_csv("MultipleLR.csv", header=None)
X = data.iloc[:, :-1].values  # Features (X1, X2, X3)
y = data.iloc[:, -1].values   # Target (Y)
```

### 3. Feature Scaling (Standardization)

```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
```

**Benefits:**
- ✔ Ensures faster and more stable convergence
- ✔ Prevents features with larger scales from dominating learning
- ✔ Improves numerical stability

### 4. Initialize Parameters

```python
m = X.shape[0]  # Number of samples
n = X.shape[1]  # Number of features

weights = np.zeros(n)  # Initialize weights
bias = 0               # Initialize bias
alpha = 0.01           # Learning rate
epochs = 1000          # Number of iterations
```

### 5. Training with Batch Gradient Descent

```python
for epoch in range(epochs):
    # Forward pass: Calculate predictions
    y_pred = np.dot(X_scaled, weights) + bias
    
    # Calculate errors
    errors = y - y_pred
    
    # Backward pass: Calculate gradients
    dw = (1/m) * np.dot(X_scaled.T, errors)
    db = (1/m) * np.sum(errors)
    
    # Update parameters
    weights += alpha * dw
    bias += alpha * db
    
    # Calculate and store loss (MSE)
    mse = np.mean(errors ** 2)
    loss_history.append(mse)
```

**Key Points:**
- Updates parameters using **all samples at once** (Batch)
- Provides **stable and smooth convergence**
- Gradient calculation: `gradient = (1/m) * X.T @ (y_pred - y)`

### 6. Model Evaluation (R-squared)

```python
# Calculate R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)        # Total sum of squares
ss_res = np.sum((y - y_pred) ** 2)        # Residual sum of squares
r2 = 1 - (ss_res / ss_tot)
```

**R² Interpretation:**
- **R² ≈ 1.0** → Excellent model (explains ~100% of variance)
- **R² ≈ 0.9** → Very good model
- **R² ≈ 0.7** → Good model
- **R² ≈ 0.5** → Fair model
- **R² ≈ 0** → Poor predictive power

### 7. Visualization

#### Perfect Fit Plot (Actual vs Predicted)

```python
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.6, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=2, label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Linear Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Sample Output Visualization:**

![Multiple Linear Regression - Perfect Fit Plot](https://github.com/mokamal11/Linear-Regression-model-using-Stochastic-Gradient-Descent/blob/main/perfect_fit_plot.png)

The plot above shows:
- 🔵 **Blue dots** = Actual vs Predicted values for each sample
- 🔴 **Red line** = Perfect fit (y = x) reference line
- Points close to the red line indicate accurate predictions
- **R² = 0.9888** indicates excellent model performance

#### Loss vs Epochs Plot

```python
plt.figure(figsize=(10, 6))
plt.plot(loss_history, lw=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📈 Sample Output

```
Dataset Information:
  - Samples: 100
  - Features: 3
  - Target: Y

Training Configuration:
  - Learning Rate (α): 0.01
  - Epochs: 1000
  - Optimizer: Batch Gradient Descent

Final Results:
  - Final Weights: [0.5847, 0.5923, 0.8534]
  - Final Bias: 0.0075
  - R-squared (R²): 0.9888
  - Final MSE: 1.2345
```

---

## 🔑 Key Mathematical Concepts

### Hypothesis Function
```
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + b
```

### Cost Function (Mean Squared Error)
```
J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
```

### Gradient Updates
```
w := w + α * (1/m) * Xᵀ * (y - ŷ)
b := b + α * (1/m) * Σ(y - ŷ)
```

Where:
- **α** = learning rate
- **m** = number of samples
- **X** = feature matrix
- **y** = actual values
- **ŷ** = predicted values

---

## 📊 Comparison: Batch vs SGD vs Mini-batch

| Aspect | Batch GD | SGD | Mini-batch GD |
|--------|----------|-----|---------------|
| **Update Frequency** | Once per epoch | Once per sample | Once per batch |
| **Computation** | Heavy | Light | Moderate |
| **Convergence** | Smooth | Noisy | Balanced |
| **Memory** | High | Low | Medium |
| **Best For** | Small datasets | Large datasets | Production |

---

## 📂 Project Structure

```
multiple-linear-regression/
├── MultipleLR.csv          # Dataset
├── linear_regression.py    # Main implementation
├── README.md               # This file
└── requirements.txt        # Dependencies
```

---

## 🔧 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multiple-linear-regression.git
cd multiple-linear-regression
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy pandas matplotlib
```

### 3. Run the Script

```bash
python linear_regression.py
```

### 4. Expected Output

- Console output with training metrics
- Perfect fit plot (Actual vs Predicted)
- Loss vs Epochs visualization

---

## 📝 Example Code Snippet

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("MultipleLR.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize features
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

# Initialize parameters
m, n = X.shape
weights = np.zeros(n)
bias = 0
alpha, epochs = 0.01, 1000

# Training loop
for epoch in range(epochs):
    y_pred = np.dot(X, weights) + bias
    errors = y - y_pred
    weights += (alpha/m) * np.dot(X.T, errors)
    bias += (alpha/m) * np.sum(errors)

# Evaluate
r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
print(f"R²: {r2:.4f}")

# Visualize
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r-')
plt.show()
```

---

## 📚 References & Resources

- [Linear Regression Tutorial](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent Explained](https://en.wikipedia.org/wiki/Gradient_descent)
- [NumPy Documentation](https://numpy.org/doc/)
- [Feature Scaling Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⭐ If you found this helpful, please give it a star!

```
⭐ Star this repository if it helped you learn!
```
