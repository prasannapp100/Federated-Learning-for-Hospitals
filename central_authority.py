import torch
import io
import logging
import copy
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import torchvision.transforms as transforms

# Setup Professional Logging
logger = logging.getLogger("uvicorn.error")
app = FastAPI()

# 1. Schema for the incoming JSON data
class HospitalUpdate(BaseModel):
    hospital_id: str
    num_samples: int
    weights_hex: str

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

# Global State
global_model = MedicalCNN()
updates_received = []
CURRENT_ROUND = 0
MIN_HOSPITALS = 2 

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/download_model")
async def download_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    logger.info(f"--- Round {CURRENT_ROUND}: Model downloaded ---")
    return {"round": CURRENT_ROUND, "weights": buffer.getvalue().hex()}

@app.get("/stats")
async def get_stats():
    return {
        "current_round": CURRENT_ROUND,
        "hospitals_connected": len(updates_received),
        "threshold": MIN_HOSPITALS
    }

@app.post("/upload_update")
async def upload_update(data: HospitalUpdate): # Use the Schema here
    global CURRENT_ROUND, updates_received
    try:
        # Extract from the JSON body
        weights_bin = bytes.fromhex(data.weights_hex)
        local_weights = torch.load(io.BytesIO(weights_bin), weights_only=True)
        
        updates_received.append({
            "weights": local_weights, 
            "n": data.num_samples, 
            "id": data.hospital_id
        })
        
        logger.info(f"SUCCESS: Received update from {data.hospital_id}")

        if len(updates_received) >= MIN_HOSPITALS:
            aggregate_and_update_global()
            return {"status": "Round Complete", "new_round": CURRENT_ROUND}
        
        return {"status": "Waiting", "received_count": len(updates_received)}
    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"status": "Error", "message": str(e)}

def aggregate_and_update_global():
    global CURRENT_ROUND, updates_received, global_model
    total_samples = sum(u['n'] for u in updates_received)
    new_weights = copy.deepcopy(updates_received[0]['weights'])
    for key in new_weights.keys():
        new_weights[key] = new_weights[key].float() * (updates_received[0]['n'] / total_samples)
        for i in range(1, len(updates_received)):
            new_weights[key] += updates_received[i]['weights'][key].float() * (updates_received[i]['n'] / total_samples)
    global_model.load_state_dict(new_weights)
    CURRENT_ROUND += 1
    updates_received = []
    logger.info(f"!!! ROUND {CURRENT_ROUND} LIVE !!!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    input_tensor = inference_transform(image).unsqueeze(0)
    global_model.eval()
    with torch.no_grad():
        output = global_model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        conf = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()
    return {"prediction": "PNEUMONIA" if prediction == 1 else "NORMAL", "confidence": f"{conf*100:.2f}%"}