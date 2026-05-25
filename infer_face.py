from ultralytics import YOLO

# 1. Load trained model (Bắt buộc dùng task="face" để model biết nó đang xử lý face object & landmarks)
model = YOLO(
    "/Users/ngoquangduc/Desktop/workspace/face_detection/ultralytics/runs/face/train12/weights/best.pt", task="face"
)

img_path = "/Users/ngoquangduc/Desktop/workspace/face_detection/WIDER_val/images/1--Handshaking/1_Handshaking_Handshaking_1_275.jpg"

# 2. Run Inference kèm than số config (thêm conf=0.1 để filter các bbox có score quá thấp)
results = model.predict(source=img_path, conf=0.1)

# Lấy kết quả ảnh đầu tiên
res = results[0]

print("\n=== KẾT QUẢ ===")
print(f"Phát hiện được {len(res.boxes)} khuôn mặt với confidence > 0.1\n")

# Vòng lặp tách toạ độ và landmarks cho từng khuôn mặt
for i, (box, conf) in enumerate(zip(res.boxes.xyxy, res.boxes.conf)):
    print(f"Face {i + 1}:")
    print(f"  BBox (x1, y1, x2, y2): {box.tolist()}")
    print(f"  Confidence: {conf.item():.4f}")
    if res.keypoints is not None:
        print(f"  Landmarks (5 điểm x,y): {res.keypoints.xy[i].tolist()}")

# 3. Vẽ visualization (Sẽ tự động draw hộp BBox & Landmark lên ảnh dựa theo param config lúc predict)
res.save("output_image.jpg")
print("Đã lưu ảnh preview tại file output_image.jpg")
