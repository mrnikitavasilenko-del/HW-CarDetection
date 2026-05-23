import cv2
from ultralytics import YOLO

# =========================
# НАСТРОЙКИ
# =========================

# Путь к входному видео с машинами
VIDEO_PATH = r"C:\Users\Nikita\Desktop\MAGA\Comp_vision\HW-CarDetection\input.mp4"

# Веса обученной модели YOLOv8 (best.pt — результат обучения на нашем датасете)
WEIGHTS = r"C:\Users\Nikita\Desktop\MAGA\Comp_vision\HW-CarDetection\Vehicle-Detection\train\weights\best.pt"

# Выходной файл (XVID работает в стандартном проигрывателе Windows)
OUTPUT_PATH = r"output_result.avi"

IMG_SIZE = 640    # размер кадра при инференсе (стандарт YOLOv8)
CONF = 0.25       # порог уверенности: боксы ниже этого значения отбрасываются

# =========================
# ЗАГРУЗКА МОДЕЛИ
# =========================

# Загружаем обученную YOLOv8 модель из файла весов
model = YOLO(WEIGHTS)

# =========================
# ОТКРЫВАЕМ ВИДЕО
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

# Читаем параметры видео для настройки VideoWriter
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Инициализируем запись результата (XVID — широко поддерживаемый кодек)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

print("🎥 Запуск обработки видео...")

# =========================
# ОСНОВНОЙ ЦИКЛ ДЕТЕКЦИИ
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break  # видео закончилось

    # Запускаем YOLOv8 на текущем кадре
    results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)

    # .plot() рисует bounding boxes и метки прямо на кадре
    annotated_frame = results[0].plot()

    # Показываем в реальном времени
    cv2.imshow("YOLO Detection", annotated_frame)

    # Записываем аннотированный кадр в файл
    out.write(annotated_frame)

    # Q — досрочный выход
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# ОСВОБОЖДАЕМ РЕСУРСЫ
# =========================

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Готово!")
print(f"📁 Сохранено в: {OUTPUT_PATH}")
