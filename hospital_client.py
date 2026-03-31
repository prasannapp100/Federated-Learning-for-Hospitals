import torch
import requests
import io
import os
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000"
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "Hospital_A") 
EPOCHS = 3
BATCH_SIZE = 8
TARGET_DIR = r"C:\FL PBL\hospital_data"
MU = 0.01 

class MedicalCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def get_local_data():
    data_path = os.path.join(TARGET_DIR, HOSPITAL_ID) 
    transform = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((64, 64)), 
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_locally(model, global_model_weights, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    global_weights_tensors = {k: v.clone().detach() for k, v in global_model_weights.items()}
    
    model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            base_loss = criterion(outputs, labels)
            proximal_term = 0.0
            for name, param in model.named_parameters():
                proximal_term += ((param - global_weights_tensors[name])**2).sum()
            loss = base_loss + (MU / 2) * proximal_term
            loss.backward()
            optimizer.step()
    return model.state_dict()

def run_fl_round():
    print(f"[{HOSPITAL_ID}] Starting Round...")
    resp = requests.get(f"{SERVER_URL}/download_model").json()
    weights_bin = bytes.fromhex(resp['weights'])
    global_weights = torch.load(io.BytesIO(weights_bin), weights_only=True)
    
    model = MedicalCNN()
    model.load_state_dict(global_weights)
    
    train_loader = get_local_data()
    new_weights = train_locally(model, global_weights, train_loader)
    
    # Upload via JSON Body
    buf = io.BytesIO()
    torch.save(new_weights, buf)
    
    payload = {
        "hospital_id": HOSPITAL_ID,
        "num_samples": len(train_loader.dataset),
        "weights_hex": buf.getvalue().hex()
    }
    
    res = requests.post(f"{SERVER_URL}/upload_update", json=payload)
    print(f"[{HOSPITAL_ID}] Server Response: {res.json()}")

if __name__ == "__main__":
    run_fl_round()