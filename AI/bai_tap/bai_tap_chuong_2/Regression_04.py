import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu giả lập
np.random.seed(42)
N = 300
X = np.linspace(-10, 10, N).reshape(-1, 1)
Y = 3 * X**2 + 2 * X + 5 + 20 * np.random.randn(N, 1)

# Biến đổi sang đa thức bậc 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Hồi quy tuyến tính trên đặc trưng bậc 2
model = LinearRegression()
model.fit(X_poly, Y)
Y_pred = model.predict(X_poly)

# Vẽ
plt.scatter(X, Y, color="salmon", s=10, alpha=0.7, label="Data")
plt.plot(X, Y_pred, color="blue", linewidth=2, label="Quadratic Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2nd Order Polynomial Regression")
plt.legend()
plt.show()
