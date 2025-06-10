import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# 1. 데이터 읽기
df = pd.read_csv('D:/workspace/arches/test/TEST_COMPANY_DATA.csv',header=None, names=['months', 'companyCode', 'sales', 'skedCount'])

# 2. 피벗 테이블: 고객별 시계열 벡터 만들기
pivot_df = df.pivot_table(index=['companyCode'], columns='months', values='sales').fillna(0)

# 3. 정규화 + 차원 축소 + 클러스터링
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(pivot_df)
pca = PCA(n_components=5)
sales_pca = pca.fit_transform(sales_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(sales_pca)
pivot_df['cluster'] = clusters
df = df.merge(pivot_df['cluster'], on=['companyCode'])

# 4. 시퀀스 생성 함수
def create_sequences(df, time_steps=3, pred_len=3):
    X, y = [], []
    grouped = df.groupby(['companyCode'])
    for _, group in grouped:
        group = group.sort_values('months')
        values = group['sales'].values
        values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        for i in range(len(values_scaled) - time_steps - pred_len + 1):
            seq = values_scaled[i:i + time_steps]
            target = values_scaled[i + time_steps:i + time_steps + pred_len]
            X.append(seq.reshape(-1, 1))
            y.append(target)
    return np.array(X), np.array(y)

# 5. 클러스터별 LSTM 학습
cluster_models = {}
for c in sorted(df['cluster'].unique()):
    print(f"\n📦 클러스터 {c} 학습 중...")

    cluster_df = df[df['cluster'] == c]
    X, y = create_sequences(cluster_df)

    if X.shape[0] == 0: continue
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    cluster_models[c] = model

# 6. 예측 예시
sample = df[(df['companyCode'] == 'CP01252')].sort_values('months')
cluster = sample['cluster'].iloc[0]
model = cluster_models[cluster]

sales = sample['sales'].values
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()
last_seq = sales_scaled[-3:].reshape(1, 3, 1)
pred = model.predict(last_seq)
predicted_sales = scaler.inverse_transform([[pred[0][0]]])[0][0]

print(f"\n📈 2025년 5월 (클러스터 {cluster}) 예측 매출: {predicted_sales:.2f}")

predicted_sales = scaler.inverse_transform([[pred[0][1]]])[0][0]

print(f"\n📈 2025년 6월 (클러스터 {cluster}) 예측 매출: {predicted_sales:.2f}")

predicted_sales = scaler.inverse_transform([[pred[0][2]]])[0][0]

print(f"\n📈 2025년 7월 (클러스터 {cluster}) 예측 매출: {predicted_sales:.2f}")




