// JavaScript for gesture recognition web app (single frame version)
class GestureWebApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;

        this.isStreaming = false;
        this.isProcessing = false;

        this.stream = null;
        this.sendInterval = null;
        this.sendFPS = 10;  // Reduced FPS since we're processing every frame

        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');

        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');

        this.fpsDisplay = document.getElementById('fps');
        this.inferenceDisplay = document.getElementById('inference');
        this.preprocessingDisplay = document.getElementById('preprocessing');

        this.gestureDisplay = document.getElementById('gesture');
        this.confidenceDisplay = document.getElementById('confidence');

        this.embeddingsList = document.getElementById('embeddingsList');

        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
    }

    async startCamera() {
        try {
            this.updateStatus("Запуск камеры...", "loading");

            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                }
            });

            this.video.srcObject = this.stream;
            this.isStreaming = true;

            this.video.addEventListener("loadedmetadata", () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.startFrameSending();  // Start sending frames immediately
            });

            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus("Camera is active", "online");

        } catch (err) {
            this.updateStatus("Camera error", "offline");
            this.showError("Could not initialize camera");
            console.error(err);
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        if (this.sendInterval) {
            clearInterval(this.sendInterval);
            this.sendInterval = null;
        }

        this.isStreaming = false;
        this.video.srcObject = null;

        this.updateStatus("Camera stopped", "offline");
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;

        this.clearResults();
    }

    startFrameSending() {
        if (this.sendInterval) return;

        const intervalMs = 1000 / this.sendFPS;

        this.sendInterval = setInterval(() => this.captureAndSendFrame(), intervalMs);
        console.log(`Sending frames every ${intervalMs} ms (${this.sendFPS} FPS)`);
    }

    async captureAndSendFrame() {
        if (!this.isStreaming || this.isProcessing) return;

        this.isProcessing = true;

        // Draw current frame
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Encode
        const frameData = this.canvas.toDataURL("image/jpeg", 0.8);

        // Send to backend
        try {
            const response = await fetch("http://localhost:8000/api/process_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: frameData })
            });

            if (!response.ok) {
                this.isProcessing = false;
                return;
            }

            const result = await response.json();
            this.handleServerResult(result);

        } catch (err) {
            console.error("Frame send error:", err);
        } finally {
            this.isProcessing = false;
        }
    }

    handleServerResult(result) {
        // Handle no hands detected case
        if (result.status === 'no_hands_detected') {
            this.gestureDisplay.textContent = "Gesture: No Hands detected";
            this.confidenceDisplay.textContent = "Accuracy: 0%";
            this.embeddingsList.innerHTML = "";
            return;
        }

        // Handle error case
        if (result.error) {
            console.error("Server error:", result.error);
            return;
        }

        // Update UI with prediction results
        if (result.gesture !== undefined) {
            this.updateUI(result);
        }
    }

    updateUI(result) {
        // Stats
        if (result.inference_time !== undefined)
            this.inferenceDisplay.textContent = result.inference_time.toFixed(1) + "ms";
        if (result.preprocessing_time !== undefined)
            this.preprocessingDisplay.textContent = result.preprocessing_time.toFixed(1) + "ms";

        // Gesture
        this.gestureDisplay.textContent = "Gesture: " + result.gesture;
        this.confidenceDisplay.textContent = "Probability: " + result.confidence.toFixed(1) + "%";

        // Predictions
        this.updatePredictions(result.embeddings || []);
    }

    updatePredictions(list) {
        this.embeddingsList.innerHTML = "";

        list.forEach(([label, conf]) => {
            const div = document.createElement("div");
            div.className = "prediction-item";
            div.innerHTML = `
                <span class="prediction-label">Жест ${label}</span>
                <span class="prediction-confidence">${conf.toFixed(1)}%</span>
            `;
            this.embeddingsList.appendChild(div);
        });
    }

    clearResults() {
        this.gestureDisplay.textContent = "Gesture: Not Detected";
        this.confidenceDisplay.textContent = "Probability: 0%";
        this.inferenceDisplay.textContent = "0ms";
        this.preprocessingDisplay.textContent = "0ms";

        this.embeddingsList.innerHTML = "";
    }

    updateStatus(text, status) {
        this.statusText.textContent = text;
        this.statusIndicator.className = `status-indicator status-${status}`;
    }

    showError(msg) {
        const div = document.createElement("div");
        div.className = "error-message";
        div.textContent = msg;

        const container = document.querySelector(".camera-section");
        container.insertBefore(div, container.firstChild);

        setTimeout(() => div.remove(), 5000);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const app = new GestureWebApp();

    if (!navigator.mediaDevices?.getUserMedia) {
        app.showError("Your browser does not support cam");
        document.getElementById("startBtn").disabled = true;
    }
});