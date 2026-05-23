from ultralytics import YOLO
from pathlib import Path
import glob
import os

# =========================
# ПУТИ ПРОЕКТА
# =========================

PROJECT_ROOT = Path(r"C:\Users\Nikita\Desktop\MAGA\Comp_vision\HW-CarDetection")

# dataset.yaml описывает структуру датасета: пути к train/val/test и имена классов
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

# Папка с изображениями датасета (train / val / test внутри)
DATASET_DIR = PROJECT_ROOT / "Vehicle-Detection" / "Dataset"

# Папка куда YOLOv8 сохраняет веса после обучения
RUNS_DIR = PROJECT_ROOT / "Vehicle-Detection" / "train"

# Тестовые изображения для демонстрации детекции после обучения
TEST_DIR = DATASET_DIR / "images" / "test"


# =========================
# ПРОВЕРКА ПУТЕЙ
# =========================

def check_paths():
    """Убеждаемся что датасет и тестовые изображения на месте перед запуском."""
    assert DATASET_YAML.exists(), f"dataset.yaml не найден: {DATASET_YAML}"
    assert TEST_DIR.exists(), f"test папка не найдена: {TEST_DIR}"
    print("✅ Пути проверены")


# =========================
# ОБУЧЕНИЕ МОДЕЛИ
# =========================

def train():
    """
    Обучаем YOLOv8n на нашем датасете транспортных средств.
    yolov8n.pt — предобученная нанo-модель (самая лёгкая, быстро обучается).
    Веса сохраняются в Vehicle-Detection/train/weights/best.pt
    """
    print("\n🚀 START TRAINING\n")

    model = YOLO("yolov8n.pt")  # загружаем базовую предобученную модель

    model.train(
        data=str(DATASET_YAML),
        epochs=10,        # количество эпох (10 — быстрое обучение для демо)
        imgsz=416,        # размер входного изображения
        batch=4,          # размер батча (маленький для слабых GPU/CPU)
        project=str(PROJECT_ROOT / "Vehicle-Detection"),
        name="train",
        exist_ok=True     # перезаписываем результаты если папка уже есть
    )


# =========================
# ПОИСК ЛУЧШИХ ВЕСОВ
# =========================

def find_best():
    """Ищем best.pt — файл с лучшими весами по итогам обучения."""
    pattern = str(RUNS_DIR / "weights" / "best.pt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"❌ best.pt не найден: {pattern}")

    print(f"\n✔ MODEL FOUND:\n{files[0]}")
    return files[0]


# =========================
# ДЕТЕКЦИЯ НА ТЕСТОВЫХ ИЗОБРАЖЕНИЯХ
# =========================

def detect(weights_path):
    """
    Запускаем детекцию на первом тестовом изображении.
    Результат (кадр с bounding boxes) сохраняется автоматически в runs/detect/.
    """
    print("\n🎯 RUN DETECTION\n")

    model = YOLO(weights_path)

    # Берём первое тестовое изображение
    images = list(TEST_DIR.glob("*.*"))
    if not images:
        raise FileNotFoundError(f"❌ Нет test изображений: {TEST_DIR}")

    img = str(images[0])
    results = model(img, save=True)  # save=True → сохраняет результат с боксами

    print("\n✅ Detection done")
    print("📁 Saved in runs/detect")


# =========================
# ПОЛНЫЙ ПАЙПЛАЙН
# =========================

if __name__ == "__main__":
    print("\n=== FINAL PIPELINE START ===\n")

    check_paths()   # 1. проверяем пути
    train()         # 2. обучаем модель
    best_model = find_best()  # 3. находим лучшие веса
    detect(best_model)        # 4. запускаем детекцию

    print("\n🎉 DONE ALL PIPELINE SUCCESSFULLY")
