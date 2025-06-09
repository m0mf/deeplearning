import pandas as pd
import numpy as np

# 1. 데이터 읽기

# data = pd.read_csv('D:/workspace/arches/test/test_data_2.csv',header=None, names=['months', 'sales', 'plan'])
data = pd.read_csv('D:/workspace/arches/test/TEST_COMPANY_DATA.csv',header=None, names=['months', 'companyCode', 'sales', 'skedCount'])

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 범주형 변수 인코딩
le = LabelEncoder()
data['months'] = le.fit_transform(data['months'])
next_date = le.transform(['2025-04'])[0]
data['companyCode'] = le.fit_transform(data['companyCode'])
companyCode = le.transform(['CP01718'])[0]
# df['region'] = le.fit_transform(df['region'])

# 피처와 타겟 분리
features = ['months', 'companyCode', 'skedCount']
target = 'sales'

X = data[features]
y = data[target]

# 훈련/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


print(X_train.shape, y_train.shape)
model = XGBRegressor(n_estimators=100, learning_rate=0.001, max_depth=5, random_state=42)
model.fit(X, y)

# 예측
print(X_val.shape)
y_pred = model.predict([[ next_date, companyCode, 0]])
print(f"다음달 매출 예상 값: {y_pred}")

# 평가
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# print(f"Validation RMSE: {rmse:.2f}")


import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()


