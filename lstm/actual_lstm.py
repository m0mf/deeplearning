import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# 1. 가상의 매출 데이터 생성 (월별 3년)

# np.random.seed(0)
# months = pd.date_range(start='2020-01', periods=36, freq='M')
# sales = np.linspace(1000, 3000, 36) + np.random.normal(0, 200, 36)
# df = pd.DataFrame({'months': months, 'sales': sales})
# print(sales)
# print(df)


# 1. csv 데이터 읽기
# data = pd.read_csv('D:/workspace/arches/test/test_data_2.csv',header=None, names=['months', 'sales', 'plan'])
# data = pd.DataFrame({'sales': data['sales'], 'plan': data['plan']})
# print(data)
data = pd.read_csv('D:/workspace/arches/test/test_data_2.csv',header=None, names=['months', 'companyCode', 'sales', 'skedCount'])

# 2. 데이터 시각화

# plt.plot(df['months'], df['sales'], label='실제 매출')
# plt.title('월별 매출 데이터')
# plt.xlabel('월')
# plt.ylabel('매출')
# plt.grid(True)
# plt.legend()
# plt.show()

# 3. 데이터 정규화(0 ~ 1 벙위)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
# 4. 시계열 데이터로 변환 (시퀀스 길이: 3)

def create_dateset(data, sql_len=3):
    X, y = [], []
    for i in range(len(data) - sql_len):
        X.append(data[i:i + sql_len])
        y.append(data[i + sql_len, 0])

    return np.array(X), np.array(y)

X,y = create_dateset(scaled_data, sql_len=3)
print(X.shape)
print(y.shape)
X = X.reshape(X.shape[0], X.shape[1], 2)
print(X.shape)

# 5. LSTM 모델 구성
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 2)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(X, y, epochs=100, verbose=0)

# 6. 다음 달 예측
last_sequence = scaled_data[-3:].reshape(1, 3, 2)
predicted_scaled = model.predict(last_sequence)
dummy_marketing = np.zeros((1, 1))
pred_combined = np.hstack([predicted_scaled, dummy_marketing])
print(pred_combined)
predicted_value = scaler.inverse_transform(pred_combined)

print(f"📊 다음 달 예상 매출: {predicted_value[0][0]:,.0f} 원")

# 7. 예측 결과 시각화
# df = pd.DataFrame({'Month': df['Month'].iloc[-1] + pd.DateOffset(months=1), 'Sales': predicted_value[0][0]})
#
# plt.plot(df['Month'], df['Sales'], marker='o', label='매출 (예측 포함)')
# plt.axvline(df['Month'].iloc[-2], color='r', linestyle='--', label='예측 시작')
# plt.title("LSTM 기반 매출 예측")
# plt.xlabel("월")
# plt.ylabel("매출")
# plt.legend()
# plt.grid(True)
# plt.show()