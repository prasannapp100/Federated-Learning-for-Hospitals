const config = window.dashboardConfig || { hospitals: [], serverOrigin: window.location.origin };

const elements = {
    apiOriginChip: document.getElementById("apiOriginChip"),
    themeToggle: document.getElementById("themeToggle"),
    roundValue: document.getElementById("roundValue"),
    roundMeta: document.getElementById("roundMeta"),
    connectedValue: document.getElementById("connectedValue"),
    connectedMeta: document.getElementById("connectedMeta"),
    modelStatusValue: document.getElementById("modelStatusValue"),
    modelStatusMeta: document.getElementById("modelStatusMeta"),
    runTrainingButton: document.getElementById("runTrainingButton"),
    serverUrlValue: document.getElementById("serverUrlValue"),
    trainingStateValue: document.getElementById("trainingStateValue"),
    trainingProgressValue: document.getElementById("trainingProgressValue"),
    trainingProgressFill: document.getElementById("trainingProgressFill"),
    hospitalGrid: document.getElementById("hospitalGrid"),
    trainingLogList: document.getElementById("trainingLogList"),
    trainingTimestamp: document.getElementById("trainingTimestamp"),
    eventList: document.getElementById("eventList"),
    xrayInput: document.getElementById("xrayInput"),
    imagePreview: document.getElementById("imagePreview"),
    previewEmpty: document.getElementById("previewEmpty"),
    predictButton: document.getElementById("predictButton"),
    predictMessage: document.getElementById("predictMessage"),
    predictionValue: document.getElementById("predictionValue"),
    confidenceValue: document.getElementById("confidenceValue"),
};

let selectedFile = null;

function formatTime(isoString) {
    if (!isoString) {
        return "Not available";
    }

    return new Date(isoString).toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
    });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dashboard-theme", theme);
}

function initializeTheme() {
    const savedTheme = localStorage.getItem("dashboard-theme");
    if (savedTheme) {
        setTheme(savedTheme);
        return;
    }

    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(prefersDark ? "dark" : "light");
}

function renderHospitals(training) {
    const items = config.hospitals.map((hospital) => {
        let state = "Pending";
        let stateClass = "";

        if (training.completed_hospitals.includes(hospital)) {
            state = "Completed";
            stateClass = "is-complete";
        } else if (training.current_hospital === hospital && training.is_running) {
            state = "In progress";
            stateClass = "is-active";
        }

        return `
            <div class="hospital-card ${stateClass}">
                <span class="hospital-name">${escapeHtml(hospital)}</span>
                <span class="hospital-state">${state}</span>
            </div>
        `;
    });

    elements.hospitalGrid.innerHTML = items.join("");
}

function renderLogs(logs) {
    if (!logs.length) {
        elements.trainingLogList.innerHTML = '<div class="empty-state compact">Training logs will appear here once a round starts.</div>';
        return;
    }

    elements.trainingLogList.innerHTML = logs
        .map(
            (entry) => `
                <div class="log-entry level-${escapeHtml(entry.level || "info")}">
                    <strong>${escapeHtml(entry.message)}</strong>
                    <time>${escapeHtml(formatTime(entry.timestamp))}</time>
                </div>
            `
        )
        .join("");
}

function renderEvents(events) {
    if (!events.length) {
        elements.eventList.innerHTML = '<div class="empty-state compact">Live backend events will populate here.</div>';
        return;
    }

    elements.eventList.innerHTML = events
        .map(
            (event) => `
                <div class="event-entry level-${escapeHtml(event.level || "info")}">
                    <strong>${escapeHtml(event.title)}</strong>
                    <span>${escapeHtml(event.detail)}</span>
                    <time>${escapeHtml(formatTime(event.timestamp))}</time>
                </div>
            `
        )
        .join("");
}

function updateStats(stats) {
    elements.roundValue.textContent = String(stats.current_round);
    elements.roundMeta.textContent =
        stats.current_round > 0
            ? "Most recent aggregation completed successfully."
            : "Waiting for first aggregation";
    elements.connectedValue.textContent = `${stats.hospitals_connected}/${stats.threshold}`;
    elements.connectedMeta.textContent =
        stats.hospitals_connected > 0
            ? "Hospital updates are flowing into the current round."
            : "No hospital updates received yet";
    elements.modelStatusValue.textContent = stats.model_status;
    elements.modelStatusMeta.textContent =
        stats.model_status === "Operational"
            ? "Global model is serving training and inference traffic."
            : "Global model is ready for the initial training cycle.";
    renderEvents(stats.recent_events || []);
}

function updateTraining(training) {
    const percent = Math.round((training.progress || 0) * 100);
    const stateLabel = training.is_running
        ? `Running${training.current_hospital ? ` - ${training.current_hospital}` : ""}`
        : training.last_error
            ? "Needs attention"
            : training.finished_at
                ? "Completed"
                : "Idle";

    elements.runTrainingButton.disabled = training.is_running;
    elements.trainingStateValue.textContent = stateLabel;
    elements.trainingProgressValue.textContent = `${percent}%`;
    elements.trainingProgressFill.style.width = `${percent}%`;
    elements.serverUrlValue.textContent = training.server_url || config.serverOrigin;
    elements.trainingTimestamp.textContent = training.started_at
        ? `Started ${formatTime(training.started_at)}`
        : "No training run started";

    renderHospitals(training);
    renderLogs(training.logs || []);

    if (training.last_error) {
        elements.trainingTimestamp.textContent = `Last error at ${formatTime(training.finished_at)}`;
    }
}

async function fetchDashboardState() {
    try {
        const [statsResponse, trainingResponse] = await Promise.all([
            fetch("/stats"),
            fetch("/api/training/status"),
        ]);

        const stats = await statsResponse.json();
        const training = await trainingResponse.json();

        updateStats(stats);
        updateTraining(training);
    } catch (error) {
        elements.connectedMeta.textContent = "Unable to reach backend.";
        elements.predictMessage.textContent = "Dashboard refresh paused until the backend is reachable.";
    }
}

async function startTraining() {
    elements.runTrainingButton.disabled = true;
    elements.trainingStateValue.textContent = "Starting";

    try {
        const response = await fetch("/api/training/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ server_url: config.serverOrigin }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Unable to start training.");
        }

        await fetchDashboardState();
    } catch (error) {
        elements.trainingStateValue.textContent = "Unable to start";
        elements.trainingTimestamp.textContent = error.message;
        elements.runTrainingButton.disabled = false;
    }
}

function handleFileSelection(file) {
    selectedFile = file || null;
    elements.predictButton.disabled = !selectedFile;
    elements.predictionValue.textContent = "Not available";
    elements.confidenceValue.textContent = "--";

    if (!selectedFile) {
        elements.imagePreview.hidden = true;
        elements.previewEmpty.hidden = false;
        elements.predictMessage.textContent = "Select an image to begin.";
        return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    elements.imagePreview.src = objectUrl;
    elements.imagePreview.hidden = false;
    elements.previewEmpty.hidden = true;
    elements.predictMessage.textContent = `${selectedFile.name} ready for inference.`;
}

async function runPrediction() {
    if (!selectedFile) {
        return;
    }

    elements.predictButton.disabled = true;
    elements.predictMessage.textContent = "Running prediction...";

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }

        const result = await response.json();
        elements.predictionValue.textContent = result.prediction;
        elements.confidenceValue.textContent = result.confidence;
        elements.predictMessage.textContent = "Prediction completed successfully.";
        await fetchDashboardState();
    } catch (error) {
        elements.predictMessage.textContent = error.message;
    } finally {
        elements.predictButton.disabled = !selectedFile;
    }
}

function bindEvents() {
    elements.themeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        setTheme(currentTheme === "dark" ? "light" : "dark");
    });

    elements.runTrainingButton.addEventListener("click", startTraining);
    elements.predictButton.addEventListener("click", runPrediction);

    elements.xrayInput.addEventListener("change", (event) => {
        handleFileSelection(event.target.files?.[0]);
    });
}

function initialize() {
    initializeTheme();
    bindEvents();

    elements.apiOriginChip.textContent = `Backend ${config.serverOrigin}`;
    elements.serverUrlValue.textContent = config.serverOrigin;

    fetchDashboardState();
    window.setInterval(fetchDashboardState, 4000);
}

initialize();
