from ultralytics import YOLO
from pathlib import Path
import glob
import os

# =========================
# АВТО-НАСТРОЙКИ ПРОЕКТА
# =========================
PROJECT_ROOT = Path(r"C:\Users\Nikita Vasilenko\Desktop\Ucheba\CV\HW-CarDetection")
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
DATASET_DIR = PROJECT_ROOT / "Vehicle-Detection" / "Dataset"

RUNS_DIR = PROJECT_ROOT / "Vehicle-Detection" / "train"

TEST_DIR = DATASET_DIR / "images" / "test"


# =========================
# ПРОВЕРКИ
# =========================
def check_paths():
    assert DATASET_YAML.exists(), f"dataset.yaml не найден: {DATASET_YAML}"
    assert TEST_DIR.exists(), f"test папка не найдена: {TEST_DIR}"
    print("✅ Пути проверены")


# =========================
# TRAIN
# =========================
def train():
    print("\n🚀 START TRAINING\n")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET_YAML),
        epochs=10,
        imgsz=416,
        batch=4,
        project=str(PROJECT_ROOT / "Vehicle-Detection"),
        name="train",
        exist_ok=True
    )


# =========================
# НАЙТИ BEST.PT
# =========================
def find_best():
    pattern = str(RUNS_DIR / "weights" / "best.pt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"❌ best.pt не найден: {pattern}")

    print(f"\n✔ MODEL FOUND:\n{files[0]}")
    return files[0]


# =========================
# DETECT
# =========================
def detect(weights_path):
    print("\n🎯 RUN DETECTION\n")

    model = YOLO(weights_path)

    images = list(TEST_DIR.glob("*.*"))

    if not images:
        raise FileNotFoundError(f"❌ Нет test изображений: {TEST_DIR}")

    img = str(images[0])

    results = model(img, save=True)

    print("\n✅ Detection done")
    print("📁 Saved in runs/detect")


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    print("\n=== FINAL PIPELINE START ===\n")

    check_paths()

    train()

    best_model = find_best()

    detect(best_model)

    print("\n🎉 DONE ALL PIPELINE SUCCESSFULLY")