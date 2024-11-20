import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. 產生300個0到1000之間的隨機數列 x
np.random.seed(42)  # 確保結果可重現
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)
# 2. 定義分類規則 y
Y = np.where((X >= 500) & (X <= 800), 1, 0).ravel()

# 3. 使用 Logistic Regression 訓練模型
logistic_model = LogisticRegression()
logistic_model.fit(X, Y)
y1_pred = logistic_model.predict(X)

# 4. 使用 SVM 訓練模型
svm_model = SVC(kernel='linear')
svm_model.fit(X, Y)
y2_pred = svm_model.predict(X)

# 5. 視覺化結果
# Logistic Regression 結果
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', label='True Labels', alpha=0.5)
plt.scatter(X, y1_pred, color='red', label='Predicted Labels (Logistic)', alpha=0.5)
plt.title('Logistic Regression Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# SVM 結果
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', label='True Labels', alpha=0.5)
plt.scatter(X, y2_pred, color='green', label='Predicted Labels (SVM)', alpha=0.5)
plt.title('SVM Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
