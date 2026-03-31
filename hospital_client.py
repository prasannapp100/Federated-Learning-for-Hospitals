<<<<<<< HEAD
import torch
import requests
import io
import os
import time
=======
import io
import os
from pathlib import Path

import requests
import torch
>>>>>>> b752926 (updated frontend (not tested))
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Configuration ---
<<<<<<< HEAD
SERVER_URL = "http://127.0.0.1:8000"
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "Hospital_A") 
EPOCHS = 3
BATCH_SIZE = 8
TARGET_DIR = r"C:\FL PBL\hospital_data"
MU = 0.01 
=======
BASE_DIR = Path(__file__).resolve().parent
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000")
HOSPITAL_ID = os.getenv("HOSPITAL_ID", "Hospital_A")
EPOCHS = 3
BATCH_SIZE = 8
TARGET_DIR = Path(os.getenv("TARGET_DIR", str(BASE_DIR / "hospital_data")))
MU = 0.01

>>>>>>> b752926 (updated frontend (not tested))

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

<<<<<<< HEAD
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
=======

def get_local_data():
    data_path = TARGET_DIR / HOSPITAL_ID
    transform = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train_locally(model, global_model_weights, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    global_weights_tensors = {
        key: value.clone().detach() for key, value in global_model_weights.items()
    }

    model.train()
    for _ in range(EPOCHS):
>>>>>>> b752926 (updated frontend (not tested))
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            base_loss = criterion(outputs, labels)
            proximal_term = 0.0
<<<<<<< HEAD
            for name, param in model.named_parameters():
                proximal_term += ((param - global_weights_tensors[name])**2).sum()
=======
            for name, parameter in model.named_parameters():
                proximal_term += ((parameter - global_weights_tensors[name]) ** 2).sum()
>>>>>>> b752926 (updated frontend (not tested))
            loss = base_loss + (MU / 2) * proximal_term
            loss.backward()
            optimizer.step()
    return model.state_dict()

<<<<<<< HEAD
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
=======

def run_fl_round():
    print(f"[{HOSPITAL_ID}] Starting Round...")
    response = requests.get(f"{SERVER_URL}/download_model", timeout=15).json()
    weights_bin = bytes.fromhex(response["weights"])
    global_weights = torch.load(io.BytesIO(weights_bin), weights_only=True)

    model = MedicalCNN()
    model.load_state_dict(global_weights)

    train_loader = get_local_data()
    new_weights = train_locally(model, global_weights, train_loader)

    buffer = io.BytesIO()
    torch.save(new_weights, buffer)

    payload = {
        "hospital_id": HOSPITAL_ID,
        "num_samples": len(train_loader.dataset),
        "weights_hex": buffer.getvalue().hex(),
    }

    result = requests.post(f"{SERVER_URL}/upload_update", json=payload, timeout=30)
    print(f"[{HOSPITAL_ID}] Server Response: {result.json()}")


if __name__ == "__main__":
    run_fl_round()
>>>>>>> b752926 (updated frontend (not tested))
