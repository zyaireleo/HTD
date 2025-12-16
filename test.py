from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 示例数据（真实标签和预测概率）
y_true = [1, 1, 0, 1, 0, 1, 1, 0]
y_scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
