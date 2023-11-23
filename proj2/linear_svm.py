"""
@Description :   Linear SVM
@Author      :   Xubo Luo 
@Time        :   2023/11/23 10:56:45
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data
import _pickle as pickle

is_eval = False

# 读取数据
X_train_scaled, X_test_scaled, y_train, y_test = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 创建SVM模型
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

if not is_eval:
    # 在训练集上拟合模型
    svm_model.fit(X_train_scaled, y_train)

    # 保存模型
    with open('linear_svm.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

else:
    # 加载模型
    with open('linear_svm.pkl', 'rb') as f:
        svm_model = pickle.load(f)
        
# 在测试集上进行预测
y_pred = svm_model.predict(X_test_scaled)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
