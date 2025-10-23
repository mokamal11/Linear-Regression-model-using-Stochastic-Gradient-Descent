import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MultipleLR.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

m, n = X.shape
weights = np.zeros(n)
bias = 0
alpha = 0.0001
epochs = 100

for epoch in range(epochs):
    for i in range(m):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights) + bias
        error = yi - y_pred
        weights += alpha * error * xi
        bias += alpha * error

print("Final weights:", weights)
print("Final bias:", bias)

y_pred_all = np.dot(X, weights) + bias

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": np.round(y_pred_all, 2)
})
print("\nComparison between actual and predicted:\n")
print(comparison.head(10))

plt.figure(figsize=(8,5))
plt.plot(y, label="Actual", marker='o')
plt.plot(y_pred_all, label="Predicted", marker='x')
plt.title("Multiple Linear Regression using SGD")
plt.xlabel("Samples")
plt.ylabel("Y values")
plt.legend()
plt.grid(True)
plt.show()