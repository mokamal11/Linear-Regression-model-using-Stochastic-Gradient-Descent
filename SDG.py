import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MultipleLR.csv", header=None)
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

# Feature Scaling
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

m, n = X.shape  
weights = np.zeros(n)  
bias = 0               
alpha = 0.01
epochs = 1000

for epoch in range(epochs):
    y_pred = np.dot(X, weights) + bias
    errors = y - y_pred
    
    weights += (alpha / m) * np.dot(X.T, errors)
    bias += (alpha / m) * np.sum(errors)
    

print("\nFinal weights:", weights)
print("Final bias:", bias)

y_pred_all = np.dot(X, weights) + bias

ss_res = np.sum((y - y_pred_all)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r2:.4f}")

# THIS IS THE LINEAR PLOT YOU WANT
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred_all, alpha=0.7, s=100, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=3, label='y = x (Perfect Fit)')
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title(f"Multiple Linear Regression\ny = w₁x₁ + w₂x₂ + w₃x₃ + b\n(R² = {r2:.4f})", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
