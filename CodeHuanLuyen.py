# python CodeHuanLuyen.py --az a_z_handwritten_data.csv --model handwriting.model
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
# Thiết lập
import matplotlib
import tensorflow as tf
matplotlib.use("Agg")

# Nhập các thư viện cần thiết
from cnn.models import ResNet                # => đảm bảo đúng cấu trúc thư mục
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import sys

# Phân tích tham số dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="Đường dẫn file A-Z dataset (.csv)")
ap.add_argument("-m", "--model", type=str, required=True, help="Tên file lưu mô hình .h5")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Tên file lưu biểu đồ huấn luyện")
args = vars(ap.parse_args())

# Kiểm tra file dữ liệu đầu vào
if not os.path.exists(args["az"]):
    print(f"[LỖI] File dataset không tồn tại: {args['az']}")
    sys.exit(1)

# Siêu tham số
EPOCHS = 10
INIT_LR = 1e-1
BS = 128

# Tải dữ liệu
print("[INFO] Đang tải dữ liệu...")
print("[INFO] Đang tải dữ liệu EMNIST...")
dataset = pd.read_csv(args["az"], header=None, nrows = 200000).values
labels = dataset[:, 0]  # Cột đầu tiên là nhãn (0-61)
data = dataset[:, 1:]   # Các cột còn lại là điểm ảnh
# (azData, azLabels) = load_az_dataset(args["az"])
# (digitsData, digitsLabels) = load_mnist_dataset()
#
# # Điều chỉnh nhãn: chữ cái +10 để tránh trùng số
# azLabels += 10
#
# # Kết hợp dữ liệu
# data = np.vstack([azData, digitsData])
# labels = np.hstack([azLabels, digitsLabels])

# Chuẩn hóa ảnh về 32x32 và scale [0,1]
def fix_emnist_image(image):
    # EMNIST lưu ảnh 1 chiều, cần đưa về 28x28
    image = image.reshape(28, 28).astype("uint8")
    # Xoay và lật để ảnh về đúng chiều đọc bình thường
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image, 1)
    #image = cv2.resize(image, (28, 28))
    return image
print("[INFO] Đang xử lý hình ảnh...")
# data = [cv2.resize(image, (28, 28)) for image in data]
# data = np.array(data, dtype="float32")
data = np.array([fix_emnist_image(img) for img in data], dtype="float32") #chuyển sang tensor
data = np.expand_dims(data, axis=-1) #thêm channel
data /= 255.0

# Chuyển nhãn sang dạng one-hot
le = LabelBinarizer()
labels = le.fit_transform(labels) #chuyêển đổi nhãn [0,0,0,1,..0]

# Tính trọng số cân bằng dữ liệu
classTotals = labels.sum(axis=0)
classWeight = {i: classTotals.max() / (c + 1e-7) for i, c in enumerate(classTotals)}

# Chia train/test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tăng cường dữ liệu
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

# Biên dịch mô hình
print("[INFO] Đang biên dịch mô hình...")
opt = tf.keras.optimizers.SGD(learning_rate=INIT_LR, momentum=0.9)
model = ResNet.build(28, 28, 1, 62, (3, 3, 3), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Huấn luyện
print("[INFO] Đang huấn luyện mạng...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1
)

# Đánh giá kết quả
labelNames = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
print("[INFO] Đang đánh giá...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Lưu mô hình
print("[INFO] Đang lưu mô hình...")
model.save(args["model"], save_format="h5")

# Vẽ biểu đồ loss/accuracy
print("[INFO] Đang lưu biểu đồ huấn luyện...")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Hiển thị một số kết quả mẫu
images = []
for i in np.random.choice(np.arange(0, len(testY)), size=49):
    probs = model.predict(np.expand_dims(testX[i], axis=0))
    pred = probs.argmax(axis=1)[0]
    label = labelNames[pred]
    correct = (pred == np.argmax(testY[i]))
    color = (0, 255, 0) if correct else (0, 0, 255)

    image = (testX[i] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96))
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    images.append(image)

montage = build_montages(images, (96, 96), (7, 7))[0]
cv2.imshow("Kết quả nhận diện ký tự", montage)
cv2.waitKey(0)
