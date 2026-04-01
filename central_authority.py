import copy
import io
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel


logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent
TRAINING_HOSPITALS = ["Hospital_A", "Hospital_B"]
EVENT_LIMIT = 10
LOG_LIMIT = 16

app = FastAPI(
    title="Federated Learning for Hospitals",
    description="Backend and dashboard for the federated learning demo.",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class HospitalUpdate(BaseModel):
    hospital_id: str
    num_samples: int
    weights_hex: str


class TrainingRequest(BaseModel):
    server_url: str | None = None


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


global_model = MedicalCNN()
updates_received = []
CURRENT_ROUND = 0
MIN_HOSPITALS = len(TRAINING_HOSPITALS)
recent_events = []

training_lock = threading.Lock()
training_state = {
    "is_running": False,
    "current_hospital": None,
    "completed_hospitals": [],
    "total_hospitals": len(TRAINING_HOSPITALS),
    "logs": [],
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "server_url": None,
}

inference_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_event(title: str, detail: str, level: str = "info") -> None:
    entry = {
        "title": title,
        "detail": detail,
        "level": level,
        "timestamp": utc_now_iso(),
    }
    recent_events.insert(0, entry)
    del recent_events[EVENT_LIMIT:]
    logger.info("%s | %s", title, detail)


def append_training_log(message: str, level: str = "info") -> None:
    with training_lock:
        training_state["logs"].insert(
            0,
            {
                "message": message,
                "level": level,
                "timestamp": utc_now_iso(),
            },
        )
        del training_state["logs"][LOG_LIMIT:]


def training_snapshot() -> dict:
    with training_lock:
        completed_hospitals = list(training_state["completed_hospitals"])
        logs = list(training_state["logs"])
        total_hospitals = training_state["total_hospitals"]
        return {
            "is_running": training_state["is_running"],
            "current_hospital": training_state["current_hospital"],
            "completed_hospitals": completed_hospitals,
            "total_hospitals": total_hospitals,
            "completed_count": len(completed_hospitals),
            "progress": (
                len(completed_hospitals) / total_hospitals if total_hospitals else 0.0
            ),
            "logs": logs,
            "started_at": training_state["started_at"],
            "finished_at": training_state["finished_at"],
            "last_error": training_state["last_error"],
            "server_url": training_state["server_url"],
        }


def reset_training_state(server_url: str) -> None:
    with training_lock:
        training_state["is_running"] = True
        training_state["current_hospital"] = None
        training_state["completed_hospitals"] = []
        training_state["logs"] = []
        training_state["started_at"] = utc_now_iso()
        training_state["finished_at"] = None
        training_state["last_error"] = None
        training_state["server_url"] = server_url


def finalize_training_run(error_message: str | None = None) -> None:
    with training_lock:
        training_state["is_running"] = False
        training_state["current_hospital"] = None
        training_state["finished_at"] = utc_now_iso()
        training_state["last_error"] = error_message


def run_training_job(server_url: str) -> None:
    reset_training_state(server_url)
    record_event("Training queued", f"Sequential run started against {server_url}.")

    try:
        for hospital_id in TRAINING_HOSPITALS:
            with training_lock:
                training_state["current_hospital"] = hospital_id

            append_training_log(f"Starting local training for {hospital_id}.")
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "hospital_client.py")],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HOSPITAL_ID": hospital_id,
                    "SERVER_URL": server_url,
                },
                check=False,
            )

            if result.stdout:
                for line in result.stdout.splitlines():
                    append_training_log(line)
            if result.stderr:
                for line in result.stderr.splitlines():
                    append_training_log(line, level="warning")

            if result.returncode != 0:
                raise RuntimeError(
                    f"{hospital_id} training exited with code {result.returncode}."
                )

            with training_lock:
                training_state["completed_hospitals"].append(hospital_id)

            append_training_log(f"{hospital_id} uploaded successfully.", level="success")

        finalize_training_run()
        record_event(
            "Training complete",
            "All hospital updates were uploaded and the round orchestration finished.",
            level="success",
        )
    except Exception as exc:
        message = str(exc)
        append_training_log(message, level="error")
        finalize_training_run(error_message=message)
        record_event("Training failed", message, level="error")


def aggregate_and_update_global() -> None:
    global CURRENT_ROUND, updates_received, global_model
    total_samples = sum(update["n"] for update in updates_received)
    new_weights = copy.deepcopy(updates_received[0]["weights"])

    for key in new_weights.keys():
        new_weights[key] = (
            new_weights[key].float() * (updates_received[0]["n"] / total_samples)
        )
        for index in range(1, len(updates_received)):
            new_weights[key] += (
                updates_received[index]["weights"][key].float()
                * (updates_received[index]["n"] / total_samples)
            )

    global_model.load_state_dict(new_weights)
    CURRENT_ROUND += 1
    round_participants = [update["id"] for update in updates_received]
    updates_received = []
    record_event(
        "Round aggregated",
        f"Round {CURRENT_ROUND} completed with {len(round_participants)} hospital updates.",
        level="success",
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": "Federated Learning Control Center",
            "hospital_count": len(TRAINING_HOSPITALS),
        },
    )


@app.get("/download_model")
async def download_model():
    buffer = io.BytesIO()
    torch.save(global_model.state_dict(), buffer)
    record_event("Model downloaded", f"Serving weights for round {CURRENT_ROUND}.")
    return {"round": CURRENT_ROUND, "weights": buffer.getvalue().hex()}


@app.get("/stats")
async def get_stats():
    return {
        "current_round": CURRENT_ROUND,
        "hospitals_connected": len(updates_received),
        "threshold": MIN_HOSPITALS,
        "progress": len(updates_received) / MIN_HOSPITALS if MIN_HOSPITALS else 0.0,
        "model_status": "Operational" if CURRENT_ROUND > 0 else "Bootstrapping",
        "recent_events": recent_events,
    }


@app.get("/api/training/status")
async def get_training_status():
    return training_snapshot()


@app.post("/api/training/run")
async def start_training_run(payload: TrainingRequest, request: Request):
    if training_snapshot()["is_running"]:
        raise HTTPException(status_code=409, detail="A training run is already active.")

    server_url = (payload.server_url or str(request.base_url)).rstrip("/")
    worker = threading.Thread(
        target=run_training_job,
        args=(server_url,),
        daemon=True,
    )
    worker.start()
    return {
        "status": "started",
        "server_url": server_url,
        "training": training_snapshot(),
    }


@app.post("/upload_update")
async def upload_update(data: HospitalUpdate):
    global updates_received
    try:
        weights_bin = bytes.fromhex(data.weights_hex)
        local_weights = torch.load(io.BytesIO(weights_bin), weights_only=True)

        updates_received.append(
            {
                "weights": local_weights,
                "n": data.num_samples,
                "id": data.hospital_id,
            }
        )

        record_event(
            "Update received",
            f"{data.hospital_id} submitted {data.num_samples} local samples.",
        )

        if len(updates_received) >= MIN_HOSPITALS:
            aggregate_and_update_global()
            return {"status": "Round Complete", "new_round": CURRENT_ROUND}

        return {"status": "Waiting", "received_count": len(updates_received)}
    except Exception as exc:
        message = str(exc)
        record_event("Upload error", message, level="error")
        return {"status": "Error", "message": message}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    input_tensor = inference_transform(image).unsqueeze(0)

    global_model.eval()
    with torch.no_grad():
        output = global_model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    diagnosis = "PNEUMONIA" if prediction == 1 else "NORMAL"
    record_event(
        "Inference completed",
        f"Prediction {diagnosis} returned with {confidence * 100:.2f}% confidence.",
    )
    return {"prediction": diagnosis, "confidence": f"{confidence * 100:.2f}%"}
