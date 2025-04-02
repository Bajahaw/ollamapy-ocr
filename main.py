import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QMimeData, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import requests
import base64
import time

class OCRWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str)
    status = pyqtSignal(str)
    log = pyqtSignal(str, bool)

    def __init__(self, image_path, model, url):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.url = url
        self._is_running = True

    def run(self):
        try:
            self.status.emit("Processing image...")
            start_time = time.time()
            
            with open(self.image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            self.log.emit(f"Image encoded in {time.time()-start_time:.2f}s", False)
            
            prompt = """Extract ALL text from this image exactly as it appears.
            Preserve original formatting, line breaks, punctuation and special characters.
            Return ONLY the extracted text with NO additional commentary."""
            
            data = {
                "model": "llama-3.2-11b-vision-preview",
                "prompt": prompt,
                "images": [encoded_image],
                "stream": False
            }
            
            self.log.emit(f"Sending to {self.model}", False)
            
            # Original Ollama API request (commented out):
            # response = requests.post(
            #     f"{self.url}/api/generate",
            #     json=data,
            #     headers={"Content-Type": "application/json"},
            #     timeout=120
            # )
            # if response.status_code == 200:
            #     result = response.json()
            #     text = result.get("response", "").strip()
            #     elapsed = time.time() - start_time
            #     self.result.emit(text)
            #     self.status.emit(f"Done in {elapsed:.2f}s")
            #     self.log.emit(f"OCR completed in {elapsed:.2f}s", False)
            #     self.log.emit(response.text, True)
            # else:
            #     error_msg = f"API error: {response.text}"
            #     self.error.emit(error_msg)
            #     self.log.emit(error_msg, True)
            
            # Groq Llama Vision API request block:

            DEMO_KEY="gsk_be...l"

            GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEMO_KEY}"  # Replace dummy-api-key with your key
            }

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }
                ],
                "model": "llama-3.2-11b-vision-preview",
                "stream": False,
            }

            response = requests.post(
                GROQ_API_URL,
                json=data,
                headers=headers,
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices")[0].get("message").get("content").strip()
                elapsed = time.time() - start_time
                self.result.emit(text)
                self.status.emit(f"Done in {elapsed:.2f}s")
                self.log.emit(f"OCR completed in {elapsed:.2f}s", False)
                self.log.emit(response.text, True)
            else:
                error_msg = f"API error (Groq): {response.text}"
                self.error.emit(error_msg)
                self.log.emit(error_msg, True)
        
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.error.emit(error_msg)
            self.log.emit(error_msg, True)
        
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False


class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern OCR with Ollama")
        self.setGeometry(100, 100, 600, 600)
        
        # Variables
        self.image_path = ""
        self.ollama_url = "http://localhost:11434"
        self.preferred_model = "gemma:2b"
        self.available_models = []
        self.worker_thread = None
        
        # Setup UI
        self.init_ui()
        self.setAcceptDrops(True)
        QTimer.singleShot(100, self.check_ollama_connection)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (single panel)
        main_layout = QVBoxLayout(central_widget)
        
        # Logo/Title
        title = QLabel("OCR Tool")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Image display - clickable
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa;
            min-height: 200px;
        """)
        self.image_label.setText("Click to select an image\nor drag & drop here\nor paste from clipboard")
        self.image_label.setMinimumHeight(250)
        self.image_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image_label.mousePressEvent = self.image_click_event
        main_layout.addWidget(self.image_label)
        
        # Image info
        self.image_info = QLabel("No image loaded")
        self.image_info.setWordWrap(True)
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_info)
        
        # Control panel (vertical layout below image)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Process button (now above and expanding)
        self.process_btn = QPushButton("Run OCR")
        self.process_btn.clicked.connect(self.run_ocr)
        self.process_btn.setStyleSheet(
            "padding: 8px; background-color: #4CAF50; color: white;"
        )
        self.process_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.process_btn.setMinimumHeight(40)
        control_layout.addWidget(self.process_btn)
        
        # Model selection (now below the button)
        model_selection = QHBoxLayout()
        # model_selection.addWidget(QLabel("Model:"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("Loading models...")
        self.model_dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        model_selection.addWidget(self.model_dropdown)
        control_layout.addLayout(model_selection)
        
        main_layout.addWidget(control_panel)
        
        # Results
        main_layout.addWidget(QLabel("Extracted Text:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)
        
        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("""
            border-top: 1px solid #ddd;
            padding: 5px;
            color: #666;
        """)
        main_layout.addWidget(self.status_bar)

    def image_click_event(self, event):
        # Handle clicks on the image area
        self.select_image()

    
    # Drag and drop support
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.load_image(file_path)
                break
    
    # Clipboard paste support
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            clipboard = QApplication.clipboard()
            mime_data = clipboard.mimeData()
            
            if mime_data.hasImage():
                qimage = clipboard.image()
                self.load_pixmap(QPixmap.fromImage(qimage))
            elif mime_data.hasUrls():
                for url in mime_data.urls():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.load_image(file_path)
                        break
    
    def load_image(self, file_path):
        self.image_path = file_path
        self.image_info.setText(file_path.split('/')[-1])
        
        # Display image
        pixmap = QPixmap(file_path)
        self.load_pixmap(pixmap)
        
        self.log(f"Loaded image: {file_path}")
        self.update_status(f"Loaded: {file_path.split('/')[-1]}")
    
    def load_pixmap(self, pixmap):
        if not pixmap.isNull():
            # Scale to fit while maintaining aspect ratio
            scaled = pixmap.scaled(
                self.image_label.width() - 20,
                self.image_label.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
    
    def check_ollama_connection(self):
        self.update_status("Connecting to Ollama...")
        self.log("Attempting to connect to Ollama...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.available_models = [model["name"] for model in models]
                self.log(f"Available models: {', '.join(self.available_models)}")
                
                self.model_dropdown.clear()
                if self.available_models:
                    self.model_dropdown.addItems(self.available_models)
                    if any(self.preferred_model in model for model in self.available_models):
                        index = self.model_dropdown.findText(self.preferred_model)
                        self.model_dropdown.setCurrentIndex(index)
                
                self.update_status(f"Connected | {len(self.available_models)} models")
            else:
                self.update_status("Connection failed", error=True)
                self.log(f"Connection failed: {response.text}", error=True)
        except Exception as e:
            self.update_status(f"Error: {str(e)}", error=True)
            self.log(f"Connection error: {str(e)}", error=True)
    
    def run_ocr(self):
        if not self.image_path:
            self.update_status("No image selected", error=True)
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
        
        if not self.model_dropdown.currentText() or self.model_dropdown.currentText() == "Loading models...":
            self.update_status("No model selected", error=True)
            QMessageBox.warning(self, "Warning", "Please select a model first")
            return
        
        self.process_btn.setEnabled(False)
        self.result_text.clear()
        self.update_status("Processing...")
        
        # Use QTimer to allow UI to update before starting OCR
        QTimer.singleShot(100, self.perform_ocr)
    
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.load_image(file_path)

    def run_ocr(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
        
        if not self.model_dropdown.currentText():
            QMessageBox.warning(self, "Warning", "Please select a model first")
            return
        
        self.process_btn.setEnabled(False)
        self.result_text.clear()
        
        # Setup worker thread
        self.worker_thread = QThread()
        self.worker = OCRWorker(
            self.image_path,
            self.model_dropdown.currentText(),
            self.ollama_url
        )
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.result.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.status.connect(self.update_status)
        self.worker.log.connect(self.log_message)
        self.worker.finished.connect(self.cleanup_thread)
        
        # Start thread
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
    
    def handle_result(self, text):
        self.result_text.setPlainText(text)
    
    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", error_msg)
        self.update_status("Error occurred", error=True)
    
    def log_message(self, message, is_error):
        timestamp = time.strftime("%H:%M:%S")
        color = "\033[91m" if is_error else ""
        reset = "\033[0m" if is_error else ""
        print(f"{color}[{timestamp}] {message}{reset}")
    
    def cleanup_thread(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.process_btn.setEnabled(True)
    
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


    def call_in_main_thread(self, func, *args):
        """Thread-safe way to update the GUI"""
        self.status_bar.parent().metaObject().invokeMethod(
            self.status_bar.parent(),
            lambda: func(*args),
            Qt.ConnectionType.QueuedConnection
        )

    def update_status(self, message, error=False):
        self.status_bar.setText(message)
        if error:
            self.status_bar.setStyleSheet("color: #d32f2f;")
        else:
            self.status_bar.setStyleSheet("color: #666;")
    
    def log(self, message, error=False):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        if error:
            print(f"\033[91m{log_msg}\033[0m")  # Red for errors
        else:
            print(log_msg)
    
    def resizeEvent(self, event):
        # Resize image when window is resized
        if hasattr(self.image_label, 'pixmap') and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            self.load_pixmap(pixmap)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = OCRApp()
    window.show()
    sys.exit(app.exec())
    
