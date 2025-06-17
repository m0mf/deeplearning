import joblib
import numpy as np
from keras.src.saving import load_model

# 1. model load
scaler = joblib.load('D:/workspace/arches/test/cnn/scaler.pkl')
model = load_model('D:/workspace/arches/test/cnn/cnn_model.h5')


# 2. 예측

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