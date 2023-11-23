"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2023/11/23 11:00:26
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
def load_data(file_path):
    data = pd.read_csv(file_path)

    # 标注连续/离散字段
    # 离散字段
    category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                    'PaymentMethod']

    # 连续字段
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # 标签
    target = 'Churn'

    # ID列
    ID_col = 'customerID'
    
    # 处理异常值
    data['TotalCharges']= data['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
    data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)
    data['TotalCharges'].fillna(data['TotalCharges'].mean())
    data['TotalCharges'] = data['TotalCharges'].fillna(0)

    data = str2float(data)

    # 划分训练集和测试集
    X = data[category_cols + numeric_cols]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def str2float(data):
    # 处理离散值    
    data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    data['Churn'].replace(to_replace='No',  value=0, inplace=True)

    data['gender'].replace(to_replace='Female', value=1, inplace=True)
    data['gender'].replace(to_replace='Male', value=0, inplace=True)

    data['Partner'].replace(to_replace='Yes', value=1, inplace=True)
    data['Partner'].replace(to_replace='No',  value=0, inplace=True)

    data['Dependents'].replace(to_replace='Yes', value=1, inplace=True)
    data['Dependents'].replace(to_replace='No',  value=0, inplace=True)

    data['PhoneService'].replace(to_replace='Yes', value=1, inplace=True)
    data['PhoneService'].replace(to_replace='No',  value=0, inplace=True)

    data['MultipleLines'].replace(to_replace='Yes', value=1, inplace=True)
    data['MultipleLines'].replace(to_replace='No',  value=0, inplace=True)
    data['MultipleLines'].replace(to_replace='No phone service',  value=2, inplace=True)

    data['InternetService'].replace(to_replace='DSL', value=1, inplace=True)
    data['InternetService'].replace(to_replace='Fiber optic',  value=2, inplace=True)
    data['InternetService'].replace(to_replace='No',  value=0, inplace=True)

    data['OnlineSecurity'].replace(to_replace='Yes', value=1, inplace=True)
    data['OnlineSecurity'].replace(to_replace='No',  value=0, inplace=True)
    data['OnlineSecurity'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['OnlineBackup'].replace(to_replace='Yes', value=1, inplace=True)
    data['OnlineBackup'].replace(to_replace='No',  value=0, inplace=True)
    data['OnlineBackup'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['DeviceProtection'].replace(to_replace='Yes', value=1, inplace=True)
    data['DeviceProtection'].replace(to_replace='No',  value=0, inplace=True)
    data['DeviceProtection'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['TechSupport'].replace(to_replace='Yes', value=1, inplace=True)
    data['TechSupport'].replace(to_replace='No',  value=0, inplace=True)
    data['TechSupport'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['StreamingTV'].replace(to_replace='Yes', value=1, inplace=True)
    data['StreamingTV'].replace(to_replace='No',  value=0, inplace=True)
    data['StreamingTV'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['StreamingMovies'].replace(to_replace='Yes', value=1, inplace=True)
    data['StreamingMovies'].replace(to_replace='No',  value=0, inplace=True)
    data['StreamingMovies'].replace(to_replace='No internet service',  value=2, inplace=True)

    data['Contract'].replace(to_replace='Month-to-month', value=0, inplace=True)
    data['Contract'].replace(to_replace='One year',  value=1, inplace=True)
    data['Contract'].replace(to_replace='Two year',  value=2, inplace=True)

    data['PaperlessBilling'].replace(to_replace='Yes', value=1, inplace=True)
    data['PaperlessBilling'].replace(to_replace='No',  value=0, inplace=True)

    data['PaymentMethod'].replace(to_replace='Electronic check', value=0, inplace=True)
    data['PaymentMethod'].replace(to_replace='Mailed check',  value=1, inplace=True)
    data['PaymentMethod'].replace(to_replace='Bank transfer (automatic)',  value=2, inplace=True)
    data['PaymentMethod'].replace(to_replace='Credit card (automatic)',  value=3, inplace=True)

    return data