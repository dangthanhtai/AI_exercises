import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Points as point  # type: ignore  # Import module Points (bạn cần có file Points.py), tạo dữ liệu giả lập đường thẳng

# Số lượng điểm dữ liệu sẽ tạo ra
N = 100  

# Tạo dữ liệu huấn luyện bằng hàm Line trong module Points
# Ở đây có thể là tạo ra 100 điểm nằm trên đường thẳng y = 1*x + 2 (hàm Line(N, a, b))
Training_Data = point.Line(N, 1, 2)

# Lấy cột đầu tiên của ma trận P làm trục X (studytime)
x = Training_Data.P[:, 0]

# Lấy cột thứ hai của ma trận P làm trục Y (score)
y = Training_Data.P[:, 1]

# Vẽ scatter plot (biểu đồ phân tán) để hiển thị dữ liệu
# c='blue': màu xanh, s=5: kích thước điểm, cmap=plt.cm.Spectral: bảng màu
plt.scatter(x[:], y[:], c='blue', s=5, cmap=plt.cm.Spectral)

# Hiển thị biểu đồ
plt.show()

# Gán x và y lần lượt thành studytime và score
studytime = x
score = y

# Tạo dictionary (từ điển) để gom dữ liệu thành cặp studytime-score
data = {
    'studytime': studytime,
    'score': score
}

# In ra dictionary để kiểm tra
print(data)

# Chuyển dictionary thành DataFrame (bảng dữ liệu dạng pandas)
points = pd.DataFrame(data)

# Đặt tên file CSV sẽ lưu
file_name = 'points_line14.csv'

# Ghi DataFrame vào file CSV, bỏ cột index mặc định của pandas
points.to_csv(file_name, index=False)

# In thông báo xác nhận đã lưu file
print(f"DataFrame written to {file_name}")
