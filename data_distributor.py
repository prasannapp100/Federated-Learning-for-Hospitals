import os
import shutil
import random

# --- Update these to your actual paths ---
BASE_SOURCE = r"C:\FL PBL\chest_xray\chest_xray\train" 
TARGET_DIR = "hospital_data"
HOSPITALS = ["Hospital_A", "Hospital_B"]
CATEGORIES = ["NORMAL", "PNEUMONIA"]

def distribute_medical_data():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    # Initialize folders
    for h in HOSPITALS:
        for c in CATEGORIES:
            os.makedirs(os.path.join(TARGET_DIR, h, c), exist_ok=True)

    for cat in CATEGORIES:
        source_path = os.path.join(BASE_SOURCE, cat)
        images = [f for f in os.listdir(source_path) if f.endswith(('.jpeg', '.jpg'))]
        random.shuffle(images)

        # Let's do a 60/40 split for a bit of imbalance
        split_idx = int(len(images) * 0.6)
        
        # Distribute to Hospital A
        for img in images[:split_idx]:
            shutil.copy(os.path.join(source_path, img), os.path.join(TARGET_DIR, "Hospital_A", cat, img))
            
        # Distribute to Hospital B
        for img in images[split_idx:]:
            shutil.copy(os.path.join(source_path, img), os.path.join(TARGET_DIR, "Hospital_B", cat, img))

    print("Data distribution complete. Hospitals are now siloed.")

if __name__ == "__main__":
    distribute_medical_data()