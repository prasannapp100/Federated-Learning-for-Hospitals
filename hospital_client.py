import torch
import requests
import io
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000"
HOSPITAL_ID = "Hospital_A"  # Change this for different instances
EPOCHS = 3
BATCH_SIZE = 8
TARGET_DIR = r"C:\FL PBL\hospital_data"

# 1. Define the same model architecture as the server
class MedicalCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 channel (Grayscale), Output: 32 filters
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # If input is 64x64, after two pools (2x2), size is 16x16
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 2) # Normal vs Pneumonia

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_local_data():
    # Points to hospital_data/Hospital_A or Hospital_B
    data_path = os.path.join(TARGET_DIR, HOSPITAL_ID) 
    
    # Medical Image Preprocessing
    # 1. Grayscale: X-rays don't need color
    # 2. Resize: Uniform 64x64 for our CNN
    # 3. Normalize: Standardizes pixel values for faster convergence
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found for {HOSPITAL_ID}. Run data_distributor.py first!")

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_locally(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def run_fl_round():
    # STEP 1: Download the Global Model
    print(f"[{HOSPITAL_ID}] Downloading global model...")
    response = requests.get(f"{SERVER_URL}/download_model").json()
    
    weights_hex = response['weights']
    weights_bin = bytes.fromhex(weights_hex)
    global_weights = torch.load(io.BytesIO(weights_bin), weights_only=False)
    
    # STEP 2: Load weights into a local model
    local_model = MedicalCNN()
    local_model.load_state_dict(global_weights)
    
    # STEP 3: Train on local X-rays
    print(f"[{HOSPITAL_ID}] Starting local training...")
    train_loader = get_local_data()
    num_samples = len(train_loader.dataset)
    new_weights = train_locally(local_model, train_loader)
    
    # STEP 4: Upload the 'Trained' weights back to Server
    print(f"[{HOSPITAL_ID}] Uploading updates...")
    buffer = io.BytesIO()
    torch.save(new_weights, buffer)
    new_weights_hex = buffer.getvalue().hex()
    
    payload = {
        "hospital_id": HOSPITAL_ID,
        "num_samples": num_samples,
        "weights_hex": new_weights_hex
    }
    
    res = requests.post(f"{SERVER_URL}/upload_update", params=payload)
    print(f"Server Response: {res.json()}")

if __name__ == "__main__":
    run_fl_round()