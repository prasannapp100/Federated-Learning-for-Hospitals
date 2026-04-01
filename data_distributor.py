import os
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_SOURCE = Path(
    os.getenv("BASE_SOURCE", str(BASE_DIR / "chest_xray" / "train"))
)
TARGET_DIR = Path(os.getenv("TARGET_DIR", str(BASE_DIR / "hospital_data")))
HOSPITALS = ["Hospital_A", "Hospital_B"]
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# --- FedProx Configuration ---
TOTAL_IMAGES_LIMIT = 100


def distribute_medical_data():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    for hospital in HOSPITALS:
        for category in CATEGORIES:
            (TARGET_DIR / hospital / category).mkdir(parents=True, exist_ok=True)

    images_per_category = TOTAL_IMAGES_LIMIT // len(CATEGORIES)

    for category in CATEGORIES:
        source_path = BASE_SOURCE / category
        all_images = [name for name in os.listdir(source_path) if name.endswith((".jpeg", ".jpg"))]
        random.shuffle(all_images)

        selected_images = all_images[:images_per_category]
        split_index = int(len(selected_images) * 0.6)

        for image_name in selected_images[:split_index]:
            shutil.copy(
                source_path / image_name,
                TARGET_DIR / "Hospital_A" / category / image_name,
            )

        for image_name in selected_images[split_index:]:
            shutil.copy(
                source_path / image_name,
                TARGET_DIR / "Hospital_B" / category / image_name,
            )

    print("Data distribution complete.")
    print(f"Total images siloed: {TOTAL_IMAGES_LIMIT}")
    print("Hospital_A: ~60 images | Hospital_B: ~40 images")


if __name__ == "__main__":
    distribute_medical_data()
