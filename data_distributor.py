import os
<<<<<<< HEAD
import shutil
import random

# --- Update these to your actual paths ---
BASE_SOURCE = r"C:\FL PBL\chest_xray\chest_xray\train" 
TARGET_DIR = "hospital_data"
=======
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BASE_SOURCE = Path(
    os.getenv("BASE_SOURCE", str(BASE_DIR / "chest_xray" / "train"))
)
TARGET_DIR = Path(os.getenv("TARGET_DIR", str(BASE_DIR / "hospital_data")))
>>>>>>> b752926 (updated frontend (not tested))
HOSPITALS = ["Hospital_A", "Hospital_B"]
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# --- FedProx Configuration ---
<<<<<<< HEAD
TOTAL_IMAGES_LIMIT = 100  # Total images across all categories and hospitals

def distribute_medical_data():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    # Initialize folders
    for h in HOSPITALS:
        for c in CATEGORIES:
            os.makedirs(os.path.join(TARGET_DIR, h, c), exist_ok=True)

    # Calculate images per category to reach exactly 100 total
    images_per_category = TOTAL_IMAGES_LIMIT // len(CATEGORIES) # 50 each

    for cat in CATEGORIES:
        source_path = os.path.join(BASE_SOURCE, cat)
        # Filter for images
        all_images = [f for f in os.listdir(source_path) if f.endswith(('.jpeg', '.jpg'))]
        random.shuffle(all_images)
        
        # Limit to 50 images for this category
        selected_images = all_images[:images_per_category]

        # Split 60/40 between Hospital A and B as requested
        split_idx = int(len(selected_images) * 0.6)
        
        # Distribute to Hospital A (approx 30 images per category)
        for img in selected_images[:split_idx]:
            shutil.copy(
                os.path.join(source_path, img), 
                os.path.join(TARGET_DIR, "Hospital_A", cat, img)
            )
            
        # Distribute to Hospital B (approx 20 images per category)
        for img in selected_images[split_idx:]:
            shutil.copy(
                os.path.join(source_path, img), 
                os.path.join(TARGET_DIR, "Hospital_B", cat, img)
            )

    print(f"Data distribution complete.")
    print(f"Total images siloed: {TOTAL_IMAGES_LIMIT}")
    print(f"Hospital_A: ~60 images | Hospital_B: ~40 images")

if __name__ == "__main__":
    distribute_medical_data()
=======
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
>>>>>>> b752926 (updated frontend (not tested))
