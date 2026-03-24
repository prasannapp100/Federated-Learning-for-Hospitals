import torch
import io
import logging
import copy
from fastapi import FastAPI
from typing import List

# 1. Setup Professional Logging
logger = logging.getLogger("uvicorn.error")
app = FastAPI()

# 2. Define the Global Model (Must match Hospital Client)
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
    

# Global State
global_model = MedicalCNN()
updates_received = []
CURRENT_ROUND = 0
MIN_HOSPITALS = 2 

@app.get("/download_model")
async def download_model():
    """Hospitals call this to get the current Global Weights."""
    buffer = io.BytesIO()
    # We use weights_only=True for security in newer Torch versions
    torch.save(global_model.state_dict(), buffer)
    
    logger.info(f"--- Round {CURRENT_ROUND}: Model downloaded by a hospital ---")
    return {
        "round": CURRENT_ROUND, 
        "weights": buffer.getvalue().hex()
    }

@app.post("/upload_update")
async def upload_update(hospital_id: str, num_samples: int, weights_hex: str):
    """Hospitals call this to send their trained local weights."""
    global CURRENT_ROUND, updates_received
    
    try:
        # Convert hex back to PyTorch weights
        weights_bin = bytes.fromhex(weights_hex)
        local_weights = torch.load(io.BytesIO(weights_bin), weights_only=True)
        
        updates_received.append({
            "weights": local_weights,
            "n": num_samples,
            "id": hospital_id
        })
        
        logger.info(f"SUCCESS: Received update from {hospital_id} ({num_samples} samples)")

        # Trigger Federated Averaging if threshold is met
        if len(updates_received) >= MIN_HOSPITALS:
            aggregate_and_update_global()
            return {"status": "Round Complete", "new_round": CURRENT_ROUND}
        
        return {"status": "Waiting for other hospitals...", "received_count": len(updates_received)}

    except Exception as e:
        logger.error(f"ERROR: Failed to process update from {hospital_id}: {e}")
        return {"status": "Error", "message": str(e)}

def aggregate_and_update_global():
    global CURRENT_ROUND, updates_received, global_model
    
    logger.info(f"!!! STARTING AGGREGATION FOR ROUND {CURRENT_ROUND} !!!")
    
    total_samples = sum(u['n'] for u in updates_received)
    
    # Start with the first contributor's weights
    new_weights = copy.deepcopy(updates_received[0]['weights'])
    
    # Federal Averaging (FedAvg) Logic
    for key in new_weights.keys():
        # Weighted contribution of hospital 1
        new_weights[key] = new_weights[key].float() * (updates_received[0]['n'] / total_samples)
        
        # Add weighted contributions of remaining hospitals
        for i in range(1, len(updates_received)):
            contribution = updates_received[i]['weights'][key].float() * (updates_received[i]['n'] / total_samples)
            new_weights[key] += contribution
    
    # Update the server's global model
    global_model.load_state_dict(new_weights)
    
    CURRENT_ROUND += 1
    updates_received = [] # Clear buffer for the next round
    logger.info(f"!!! ROUND {CURRENT_ROUND} IS NOW LIVE !!!")