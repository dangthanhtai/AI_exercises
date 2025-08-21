import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Số lượng điểm dữ liệu
N = 100

# Tạo mảng x, y
x = np.zeros(N)
y = np.zeros(N)

# Tham số đường thẳng: y = ax + b + noise
a = 5 * np.random.randn(1)   # hệ số góc
b = 50 * np.random.randn(1)  # hệ số tự do
VarX = 0.5
VarY = 15

# Sinh dữ liệu ngẫu nhiên
for i in range(N):
    x[i] = i
    y[i] = a * x[i] + b + VarY * np.random.randn(1)

# Vẽ scatter plot
plt.scatter(x, y, c='blue', s=5, cmap=plt.cm.Spectral)
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Generated Data")
plt.show()

# Gán dữ liệu thành biến studytime, score
studytime = x
score = y

# Đưa vào DataFrame
data = {
    'studytime': studytime,
    'score': score
}
points = pd.DataFrame(data)

# Lưu ra file CSV
file_name = 'points_line14.csv'
points.to_csv(file_name, index=False)

print(data)
print(f"DataFrame written to {file_name}")
