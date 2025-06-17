import numpy as np
import pandas as pd
from keras.src.layers import ELU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib


# 최근 6개월 실적, 거래 횟수 데이터
n_features = 2     # 거래 횟수, 실적
n_months = 6       # 최근 6개월

# data = pd.read_csv('D:/workspace/arches/test/cnn_test_data2.csv',header=None, names=['companyCode', 'rn', 'frequency', 'sales'])
# data = pd.read_csv('D:/workspace/arches/test/cnn_test_data3.csv',header=None, names=['companyCode', 'rn', 'frequency', 'sales', 'days'])
data = pd.read_csv('D:/workspace/arches/test/cnn_test_data4.csv',header=None, names=['companyCode', 'rn', 'frequency', 'sales', 'days'])

# 정규화 - 특성별 영향도 반영을 잘하기 위해 정규화
# scaler = StandardScaler() # (평균 0, 표준 편차 1)의 정규 분포로 정규화(x - 평균 / 표준 편차)
scaler = MinMaxScaler() # 비율 값으로 계산
data['frequency'] = scaler.fit_transform(data['frequency'].values.reshape(-1, 1))

# sales => 고객별 현재값/최대값 으로 비율 값으로 계산
# data['sales'] = scaler.fit_transform(data['sales'].values.reshape(-1, 1))

# days => 202401 기준 으로 (거래일 - 202401) / (현재 - 202401) 으로 비율 값으로 계산
# data['days'] = scaler.fit_transform(data['days'].values.reshape(-1, 1))

pivot_df = data.pivot_table(index='companyCode', columns='rn', values=['frequency', 'sales', 'days']).fillna(0)

# 현재 시점 기준 최근 3개월 이상 거래가 없는 거래선 데이터
churning = pd.read_csv('D:/workspace/arches/test/cnn_one_test_data.csv',header=None, names=['companyCode', 'leave'])
churning['leave'] = 1


# CNN 입력 형태로 reshape: (샘플수, 높이=feature수, 너비=시간, 채널=1)
X_cnn = pivot_df[['frequency', 'sales', 'days']].values.reshape(-1, 3, 6, 1)
y =[]
for key in pivot_df.index:
    leave = 1 if len(churning[churning['companyCode'] == key]) != 0 else 0
    y.append([leave])

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(np.array(X_cnn), np.array(y), test_size=0.2, random_state=42)


# 2. 모델 생성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# kernel size는 input shape 을 확인 해서 설정 해야 함 kernel size <= input shape의 (height * width)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(3, 6, 1), strides=(1,3), padding='same'),  # 입력: 3x6
    MaxPooling2D(pool_size=(1, 2)),  # 시간 축(pooling 너비) 축소
    Dropout(0.2),

    Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(3, 1)),
    Dropout(0.2),

    Flatten(),  # 2D → 1D
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # 이진 분류
])

# 3. 모델 학습 및 평가

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 'mse'(수치 예측) 또는 'binary_crossentropy'(0 - 1 이진 분류), 'categorical_crossentropy'(다중 클래스 분류)
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

# epochs=100 설정 하고 데이터 셋이 작기 때문에 과적합을 방지 하기 위해서
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])


# 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {acc:.2f}")



# 7. 모델 저장
os.makedirs('D:/workspace/arches/test/cnn', exist_ok=True)
joblib.dump(scaler, 'D:/workspace/arches/test/cnn/scaler.pkl')
model.save('D:/workspace/arches/test/cnn/cnn_model.h5')


# 4. 예측

x_sample1 = np.array(
    [
        scaler.transform(np.array([0, 0, 0, 0, 0, 1]).reshape(-1, 1)),
        np.array([0, 0, 0, 0, 0, 0.034]).reshape(-1, 1),
        np.array([0, 0, 0, 0, 0, 0.116]).reshape(-1, 1),
    ]).reshape(-1,3,6,1) # 이탈 고객

x_sample2 = np.array(
    [
        scaler.transform(np.array([0, 0, 0, 0, 0, 1]).reshape(-1, 1)),
        np.array([0, 0, 0, 0, 0, 1]).reshape(-1, 1),
        np.array([0, 0, 0, 0, 0, 0.94]).reshape(-1, 1),
    ]).reshape(-1,3,6,1) # 잔류 고객

x_sample3 = np.array([
    scaler.transform(np.array([0, 1, 1, 1, 1, 1]).reshape(-1, 1)),
    np.array([0, 0.764, 0.865, 0.885, 0.658, 1]).reshape(-1, 1),
    np.array([0, 0.708, 0.768, 0.822, 0.882, 0.94]).reshape(-1, 1),
]).reshape(-1,3,6,1) # 잔류 고객

x_sample4 = np.array([
    scaler.transform(np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)),
    np.array([0, 0, 0, 0.845, 1, 0.817]).reshape(-1, 1),
    np.array([0, 0, 0, 0.294, 0.648, 0.768]).reshape(-1, 1),
]).reshape(-1,3,6,1) # 거래 주기가 긴 고객 -> 2024-06, 2024-12, 2025-02 거래 -> 잔류 고객

x_sample5 = np.array([
    scaler.transform(np.array([0, 0, 1, 1, 1, 1]).reshape(-1, 1)),
    np.array([0, 0, 0.5, 1, 0.148, 0.108]).reshape(-1, 1),
    np.array([0, 0, 0.59, 0.648, 0.708, 0.768]).reshape(-1, 1),
]).reshape(-1,3,6,1) # 이탈 고객

y_pred_proba1 = model.predict(x_sample1)
y_pred_proba2 = model.predict(x_sample2)
y_pred_proba3 = model.predict(x_sample3)
y_pred_proba4 = model.predict(x_sample4)
y_pred_proba5 = model.predict(x_sample5)

print(f'1번 이탈 고객 => 예측값: {y_pred_proba1}')
print(f'2번 잔류 고객 => 예측값: {y_pred_proba2}')
print(f'3번 잔류 고객 => 예측값: {y_pred_proba3}')
print(f'4번 잔류 고객 => 예측값: {y_pred_proba4}')
print(f'5번 이탈 고객 => 예측값: {y_pred_proba5}')

