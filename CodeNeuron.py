import tkinter as tk
import numpy as np
import cv2
import imutils
from PIL import Image, ImageTk
from tkinter import filedialog
from keras.models import load_model
from imutils.contours import sort_contours

# Nạp mô hình AI đã được huấn luyện (nhận diện 62 ký tự)
model = load_model('handwriting_62.h5')


# ==========================================
# HÀM 1: CHỌN ẢNH TỪ MÁY TÍNH
# ==========================================
def select_image():
    # Mở hộp thoại chọn file
    file_path = filedialog.askopenfilename()
    if file_path:
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        # Kiểm tra xem file chọn có đúng định dạng ảnh không
        if any(file_path.lower().endswith(fmt) for fmt in valid_formats):
            img = Image.open(file_path)
            # Thu nhỏ ảnh lại để hiển thị vừa vặn trên giao diện (không làm đổi ảnh gốc)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)

            # Cập nhật ảnh lên giao diện Tkinter
            panel.config(image=img)
            panel.image = img
            panel.file_path = file_path  # Lưu lại đường dẫn để dùng cho hàm nhận diện
        else:
            print("Vui lòng chọn ảnh có định dạng hợp lệ (.jpg, .jpeg, .png, .bmp, .gif).")


# ==========================================
# HÀM 2: XỬ LÝ ẢNH VÀ NHẬN DIỆN KÝ TỰ (CORE)
# ==========================================
def predict_image():
    # Kiểm tra xem người dùng đã chọn ảnh chưa
    if not hasattr(panel, 'file_path') or not panel.file_path:
        print("Vui lòng chọn ảnh trước khi nhận diện.")
        return
    file_path = panel.file_path
    if file_path:
        # 1. ĐỌC VÀ CHUYỂN ĐỔI ẢNH CƠ BẢN
        image = cv2.imread(file_path)  # Đọc ảnh gốc bằng OpenCV
        original_image = image.copy()  # Tạo một bản sao để lát nữa vẽ khung xanh lên
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang thang độ xám (đen trắng)

        # 2. TIỀN XỬ LÝ ẢNH (BƯỚC QUAN TRỌNG NHẤT)
        # Làm mờ nhẹ ảnh để xóa bớt các hạt nhiễu (bụi giấy, vân giấy)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Threshold: Tự động phân ngưỡng để tách nét mực khỏi nền giấy.
        # Rất hiệu quả khi ảnh bị đổ bóng hoặc ánh sáng không đều.
        # Chuyển nét chữ thành màu TRẮNG, nền thành màu ĐEN.
        thresh_full = cv2.adaptiveThreshold(blurred, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)

        # Dilation (Phép giãn): Làm cho các nét mực trắng "nở" ra và dày lên.
        # Giúp nối liền các nét chữ bị đứt đoạn (như chữ 'a' viết mỏng).
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_full = cv2.dilate(thresh_full, kernel, iterations=2)

        # 3. TÌM KIẾM KÝ TỰ (CONTOURS)
        # Tìm tất cả các khối màu trắng (chữ cái) trên nền đen
        cnts = cv2.findContours(thresh_full.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Nếu bức ảnh trống trơn, không có chữ thì dừng lại
        if len(cnts) == 0:
            print("[INFO] Không tìm thấy ký tự nào.")
            display_predicted_image(original_image)
            return

        # Sắp xếp các ký tự được tìm thấy theo thứ tự từ Trái sang Phải (để đọc đúng từ)
        cnts = sort_contours(cnts, method="left-to-right")[0]

        chars = []  # Danh sách lưu trữ các bức ảnh chữ cái nhỏ đã được cắt

        # 4. CẮT VÀ CHUẨN HÓA TỪNG KÝ TỰ (ĐỂ GIỐNG DỮ LIỆU EMNIST)
        for c in cnts:
            # Lấy tọa độ (x, y) và kích thước (w - rộng, h - cao) của khung bao quanh chữ
            (x, y, w, h) = cv2.boundingRect(c)
            # LỌC KÍCH THƯỚC: Bỏ qua các đốm nhiễu quá nhỏ (w<5 hoặc h<15)
            if (w >= 5 and w <= 300) and (h >= 15 and h <= 300):
                # Cắt lấy vùng ảnh chỉ chứa ký tự đó
                roi = thresh_full[y:y + h, x:x + w]

                # Căn chỉnh lại tỷ lệ: Ép chiều dài nhất (rộng hoặc cao) về 22 pixel
                if w > h:
                    roi = imutils.resize(roi, width=22)
                else:
                    roi = imutils.resize(roi, height=22)

                # Tính toán kích thước còn thiếu để bù viền đen sao cho đủ 28x28
                (tH, tW) = roi.shape
                dX = int(max(0, 28 - tW) / 2.0)
                dY = int(max(0, 28 - tH) / 2.0)

                # Thêm viền đen xung quanh để ảnh đạt đúng chuẩn của EMNIST
                padded = cv2.copyMakeBorder(roi, top=dY, bottom=dY,
                                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # Ép kích thước lần cuối để chắc chắn ảnh là 28x28 pixel
                padded = cv2.resize(padded, (28, 28))
                # Chuẩn hóa dữ liệu ảnh (đưa giá trị pixel từ 0-255 về 0.0-1.0 cho AI dễ tính toán)
                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis=-1)  # Thêm chiều kênh màu (kênh 1 cho ảnh xám)

                # Lưu ảnh đã xử lý và tọa độ của nó vào danh sách
                chars.append((padded, (x, y, w, h)))

        # Lấy tọa độ của tất cả các ký tự hợp lệ
        boxes = [b[1] for b in chars]
        # Chuyển danh sách ảnh thành mảng Numpy để đưa vào Keras
        chars = np.array([c[0] for c in chars], dtype="float32")

        # 5. MÔ HÌNH AI DỰ ĐOÁN
        preds = model.predict(chars)

        # Định nghĩa 62 nhãn tương ứng với 62 lớp đầu ra của mô hình
        labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        labelNames = [l for l in labelNames]

        # 6. HIỂN THỊ KẾT QUẢ
        # Lặp qua từng kết quả dự đoán và tọa độ tương ứng
        for (pred, (x, y, w, h)) in zip(preds, boxes):
            i = np.argmax(pred)  # Tìm vị trí có xác suất cao nhất
            prob = pred[i]  # Lấy giá trị xác suất (độ tự tin)
            label = labelNames[i]  # Suy ra ký tự tương ứng

            # Chỉ hiển thị kết quả nếu AI chắc chắn hơn 50%
            if prob > 0.4:
                print("[INFO] {} - {:.2f}%".format(label, prob * 100))
                # Vẽ hình chữ nhật màu xanh lá cây bao quanh ký tự trên ảnh gốc
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Viết ký tự dự đoán được lên trên khung xanh
                cv2.putText(original_image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Gọi hàm hiển thị ảnh gốc đã được vẽ khung
        display_predicted_image(original_image)


# ==========================================
# HÀM 3: HIỂN THỊ ẢNH KẾT QUẢ CUỐI CÙNG
# ==========================================
def display_predicted_image(result_image):
    cv2.imshow("Nhận Diện Kí Tự", result_image)
    cv2.waitKey(0)  # Đợi người dùng nhấn phím bất kỳ để đóng ảnh


# ==========================================
# CẤU TRÚC GIAO DIỆN TKINTER
# ==========================================
root = tk.Tk()
root.title("Nhận Diện Kí Tự")
root.geometry('450x450')

# Nút "Chọn Ảnh"
select_button = tk.Button(root, text="Chọn Ảnh", command=select_image)
select_button.pack(padx=20, pady=10)

# Khung chứa ảnh hiển thị
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Nút "Nhận Diện"
predict_button = tk.Button(root, text="Nhận Diện", command=predict_image)
predict_button.pack(padx=20, pady=10)

# Chạy vòng lặp chính của giao diện
root.mainloop()