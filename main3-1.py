import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. 產生300個0~1000之間的隨機數列 x
np.random.seed(42)  # 確保隨機結果可重現
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)

# 2. 根據條件定義 y
Y = np.where((X >= 500) & (X <= 800), 1, 0).ravel()

# 3. 使用 Logistic Regression 訓練模型
logistic_model = LogisticRegression()
logistic_model.fit(X, Y)
y1_pred = logistic_model.predict(X)

# 4. 使用 SVM with RBF kernel 訓練模型
svm_model_rbf = SVC(kernel='rbf', probability=True)
svm_model_rbf.fit(X, Y)
y2_pred_rbf = svm_model_rbf.predict(X)

# 5. 決策邊界可視化
# 在 0 到 1000 範圍內生成細粒度數據
x_range = np.linspace(0, 1000, 1000).reshape(-1, 1)
y1_decision = logistic_model.predict(x_range)
y2_decision = svm_model_rbf.predict(x_range)

plt.figure(figsize=(12, 6))

# Logistic Regression 決策邊界
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='True Y', alpha=0.6)
plt.scatter(X, y1_pred, color='red', label='Logistic Regression Y1', alpha=0.6)
plt.plot(x_range, y1_decision, color='black', linestyle='--', label='Decision Boundary')
plt.title('Logistic Regression Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# SVM (RBF Kernel) 決策邊界
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', label='True Y', alpha=0.6)
plt.scatter(X, y2_pred_rbf, color='green', label='SVM (RBF Kernel) Y2', alpha=0.6)
plt.plot(x_range, y2_decision, color='black', linestyle='--', label='Decision Boundary')
plt.title('SVM with RBF Kernel Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()
