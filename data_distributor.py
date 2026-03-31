import os
import shutil
import random

# --- Update these to your actual paths ---
BASE_SOURCE = r"C:\FL PBL\chest_xray\chest_xray\train" 
TARGET_DIR = "hospital_data"
HOSPITALS = ["Hospital_A", "Hospital_B"]
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# --- FedProx Configuration ---
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