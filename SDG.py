import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ تحميل البيانات
data = pd.read_csv("MultipleLR.csv", header=None) 
X = data.iloc[:, :-1].values   # الثلاث أعمدة الأولى (X1, X2, X3)
y = data.iloc[:, -1].values    # العمود الأخير هو الهدف Y

# 2️⃣ تهيئة القيم
m, n = X.shape  # m = عدد الصفوف, n = عدد المتغيرات
weights = np.zeros(n)  # الأوزان (w1, w2, w3)
bias = 0               # التقاطع (b)
alpha = 0.000001         # معدل التعلم
epochs = 1000       # عدد التكرارات

# 3️⃣ تنفيذ خوارزمية Stochastic Gradient Descent
for epoch in range(epochs):
    for i in range(m):
        xi = X[i]
        yi = y[i]

        # التنبؤ الحالي
        y_pred = np.dot(xi, weights) + bias

        # حساب الخطأ
        error = yi - y_pred

        # تحديث الأوزان والـ bias
        weights += alpha * error * xi
        bias += alpha * error

# 4️⃣ عرض النتائج النهائية
print("Final weights:", weights)
print("Final bias:", bias)

# 5️⃣ اختبار النموذج على نفس البيانات (للتوضيح)
y_pred_all = np.dot(X, weights) + bias

# 6️⃣ عرض مقارنة بسيطة بين القيم الحقيقية والمتوقعة
comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": np.round(y_pred_all, 2)
})
print("\nComparison between actual and predicted:\n")
print(comparison.head(10))

# 7️⃣ رسم مقارنة بسيطة للقيم
plt.figure(figsize=(8,5))
plt.plot(y, label="Actual", marker='o')
plt.plot(y_pred_all, label="Predicted", marker='x')
plt.title("Multiple Linear Regression using SGD")
plt.xlabel("Samples")
plt.ylabel("Y values")
plt.legend()
plt.grid(True)
plt.show()