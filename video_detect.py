import cv2
from ultralytics import YOLO

# =========================
# НАСТРОЙКИ
# =========================

VIDEO_PATH = r"C:\Users\Nikita Vasilenko\Desktop\Ucheba\CV\HW-CarDetection\input.mp4"
WEIGHTS = r"C:\Users\Nikita Vasilenko\Desktop\Ucheba\CV\HW-CarDetection\Vehicle-Detection\train\weights\best.pt"
OUTPUT_PATH = r"output_result.mp4"

IMG_SIZE = 640
CONF = 0.25

# =========================
# ЗАГРУЗКА МОДЕЛИ
# =========================

model = YOLO(WEIGHTS)

# =========================
# ВИДЕО
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# кодек для сохранения
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print("🎥 Запуск обработки видео...")

# =========================
# LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inference
    results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)

    # рисуем bounding boxes
    annotated_frame = results[0].plot()

    # показываем
    cv2.imshow("YOLO Detection", annotated_frame)

    # сохраняем
    out.write(annotated_frame)

    # выход по Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Готово!")
print(f"📁 Сохранено в: {OUTPUT_PATH}")