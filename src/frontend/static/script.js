// JavaScript для веб-приложения распознавания жестов
class GestureWebApp {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isStreaming = false;
        this.isProcessing = false;
        this.stream = null;
        this.animationId = null;
        
        this.initializeElements();
        this.setupEventListeners();
    }
    
    initializeElements() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Элементы интерфейса
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
            this.updateStatus('Запуск камеры...', 'loading');
            
            // Запрос доступа к камере
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.isStreaming = true;
            
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.startProcessing();
            });
            
            this.updateStatus('Камера активна', 'online');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
        } catch (error) {
            console.error('Ошибка при запуске камеры:', error);
            this.updateStatus('Ошибка доступа к камере', 'offline');
            this.showError('Не удалось получить доступ к камере. Проверьте разрешения.');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.isStreaming = false;
        this.isProcessing = false;
        this.video.srcObject = null;
        
        this.updateStatus('Камера остановлена', 'offline');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Очистка результатов
        this.clearResults();
    }
    
    startProcessing() {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.processFrame();
    }
    
    async processFrame() {
        if (!this.isStreaming || !this.isProcessing) return;
        
        try {
            // Захват кадра
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            const frameData = this.canvas.toDataURL('image/jpeg', 0.8);
            
            // Отправка на сервер для обработки
            const response = await fetch('/api/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame: frameData })
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateUI(result);
            } else {
                console.error('Ошибка обработки кадра:', response.statusText);
            }
            
        } catch (error) {
            console.error('Ошибка при обработке кадра:', error);
        }
        
        // Продолжение обработки
        this.animationId = requestAnimationFrame(() => this.processFrame());
    }
    
    updateUI(result) {
        // Обновление статистики
        if (result.fps !== undefined) {
            this.fpsDisplay.textContent = result.fps;
        }
        
        if (result.inference_time !== undefined) {
            this.inferenceDisplay.textContent = `${result.inference_time.toFixed(1)}ms`;
        }
        
        if (result.preprocessing_time !== undefined) {
            this.preprocessingDisplay.textContent = `${result.preprocessing_time.toFixed(1)}ms`;
        }
        
        // Обновление результатов распознавания
        if (result.gesture !== null && result.gesture !== undefined) {
            this.gestureDisplay.textContent = `Символ: ${result.gesture}`;
            this.confidenceDisplay.textContent = `Уверенность: ${result.confidence.toFixed(1)}%`;
            
            // Обновление списка предсказаний
            this.updateEmbeddings(result.embeddings);
        } else {
            this.gestureDisplay.textContent = 'Символ: Не обнаружен';
            this.confidenceDisplay.textContent = 'Уверенность: 0%';
            this.clearEmbeddings();
        }
    }
    
    updateEmbeddings(embeddings) {
        this.embeddingsList.innerHTML = '';
        
        embeddings.forEach(([char, prob]) => {
            const item = document.createElement('div');
            item.className = 'embedding-item';
            item.innerHTML = `
                <span class="embedding-char">${char}</span>
                <span class="embedding-prob">${(prob * 100).toFixed(1)}%</span>
            `;
            this.embeddingsList.appendChild(item);
        });
    }
    
    clearEmbeddings() {
        this.embeddingsList.innerHTML = '';
    }
    
    clearResults() {
        this.gestureDisplay.textContent = 'Символ: Не обнаружен';
        this.confidenceDisplay.textContent = 'Уверенность: 0%';
        this.fpsDisplay.textContent = '0';
        this.inferenceDisplay.textContent = '0ms';
        this.preprocessingDisplay.textContent = '0ms';
        this.clearEmbeddings();
    }
    
    updateStatus(message, status) {
        this.statusText.textContent = message;
        this.statusIndicator.className = `status-indicator status-${status}`;
    }
    
    showError(message) {
        // Создание элемента ошибки
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Вставка в начало контейнера
        const container = document.querySelector('.camera-section');
        container.insertBefore(errorDiv, container.firstChild);
        
        // Автоматическое удаление через 5 секунд
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Инициализация приложения при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const app = new GestureWebApp();
    
    // Проверка поддержки getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        app.showError('Ваш браузер не поддерживает доступ к камере. Пожалуйста, используйте современный браузер.');
        document.getElementById('startBtn').disabled = true;
    }
});