import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('points_line.csv')
print(data.head())

# Chuyển sang NumPy array
studytime = data['studytime'].to_numpy()
score = data['score'].to_numpy()
print(studytime)
print(score)

# Hàm tính loss (MSE)
def loss(m, b, X, Y):
    total_error = 0
    for i in range(len(X)):
        total_error += (Y[i] - (m * X[i] + b)) ** 2
    total_error = total_error / float(len(X))
    return total_error

# Hàm Gradient Descent
def gradient_descent(a_now, b_now, X, Y, LR):
    a_gradient = 0
    b_gradient = 0
    n = len(X)
    for i in range(n):
        a_gradient += -(2/n) * X[i] * (Y[i] - (a_now * X[i] + b_now))
        b_gradient += -(2/n) * (Y[i] - (a_now * X[i] + b_now))
    a = a_now - a_gradient * LR
    b = b_now - b_gradient * LR
    return a, b

# Chạy thuật toán
a = 0
b = 0
LR = 0.0003   # learning rate
epochs = 40000

for i in range(epochs):
    if (i % 1000 == 0):
        loss_val = loss(a, b, studytime, score)
        print(f'Epochs: {i}, Loss: {loss_val:.2f}')
    a, b = gradient_descent(a, b, studytime, score, LR)

print(f'The estimated line equation is: y = {a:.3f}x + {b:.3f}')

# Vẽ dữ liệu gốc và đường hồi quy
plt.scatter(studytime, score, c='blue', s=5, cmap=plt.cm.Spectral, label="Data")
plt.plot(studytime, a * studytime + b, color='red', label="Fitted Line")
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
