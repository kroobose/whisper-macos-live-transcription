"""
Whisper Mac Recording App - Desktop Version

A PyQt6 desktop application for audio transcription using OpenAI Whisper.
Features:
- Model selection (tiny, base, small, medium, large) with download progress
- Audio recording via microphone with start/stop buttons
- Real-time transcription mode
- Optional output folder for saving transcriptions
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import whisper
import torch


def get_device() -> str:
    """Get the best available device (MPS for Mac, CUDA for Nvidia, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ModelLoaderThread(QThread):
    """Thread for loading Whisper model with progress updates."""

    progress = pyqtSignal(str)
    finished_loading = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_name: str, device: str):
        super().__init__()
        self.model_name = model_name
        self.device = device

    def run(self):
        try:
            self.progress.emit(f"Loading model '{self.model_name}' on {self.device.upper()}...")
            model = whisper.load_model(self.model_name, device=self.device)
            self.progress.emit(f"Model '{self.model_name}' loaded on {self.device.upper()}!")
            self.finished_loading.emit(model)
        except Exception as e:
            self.error.emit(str(e))


class TranscribeThread(QThread):
    """Thread for transcribing audio."""

    progress = pyqtSignal(str)
    finished_transcribe = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, audio_path: str, language: str):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        self.language = language

    def run(self):
        try:
            self.progress.emit("Transcribing audio...")
            options = {"language": self.language} if self.language != "auto" else {}
            # Use fp16 on MPS/CUDA for better performance
            options["fp16"] = (self.model.device.type != "cpu")
            result = self.model.transcribe(self.audio_path, **options)
            self.progress.emit("Transcription complete!")
            self.finished_transcribe.emit(result["text"])
        except Exception as e:
            self.error.emit(str(e))


class RealtimeTranscribeThread(QThread):
    """Thread for real-time transcription of audio chunks."""

    transcription_update = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, audio_data: np.ndarray, sample_rate: int, language: str):
        super().__init__()
        self.model = model
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.language = language

    def run(self):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                audio_int16 = (self.audio_data * 32767).astype(np.int16)
                wavfile.write(temp_path, self.sample_rate, audio_int16)

            # Transcribe
            options = {"language": self.language} if self.language != "auto" else {}
            options["fp16"] = (self.model.device.type != "cpu")
            result = self.model.transcribe(temp_path, **options)
            text = result["text"].strip()

            if text:
                self.transcription_update.emit(text)

            # Cleanup
            os.unlink(temp_path)
        except Exception as e:
            self.error.emit(str(e))


class AudioRecorder:
    """Audio recorder using sounddevice with real-time support."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data: list = []
        self.realtime_buffer: list = []
        self.stream = None

    def start(self, realtime_callback=None):
        """Start recording audio."""
        self.recording = True
        self.audio_data = []
        self.realtime_buffer = []
        self.realtime_callback = realtime_callback

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                if self.realtime_callback:
                    self.realtime_buffer.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=callback
        )
        self.stream.start()

    def get_realtime_chunk(self) -> Optional[np.ndarray]:
        """Get accumulated audio for real-time processing and clear buffer."""
        if self.realtime_buffer:
            chunk = np.concatenate(self.realtime_buffer, axis=0)
            self.realtime_buffer = []
            return chunk
        return None

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            return np.concatenate(self.audio_data, axis=0)
        return np.array([])

    def save(self, filepath: str, audio_data: np.ndarray):
        """Save audio data to WAV file."""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, audio_int16)


class WhisperApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.model: Optional[whisper.Whisper] = None
        self.recorder = AudioRecorder()
        self.temp_audio_path = "/tmp/whisper_recording.wav"
        self.is_recording = False
        self.realtime_timer: Optional[QTimer] = None
        self.realtime_transcription = ""
        self.pending_transcription = False

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Whisper Transcription")
        self.setMinimumSize(650, 550)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("🎙️ Whisper Transcription")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setMinimumWidth(80)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        layout.addLayout(model_layout)

        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setMinimumWidth(80)
        self.device_combo = QComboBox()
        # Add available devices
        self.device_combo.addItem("CPU", "cpu")
        if torch.backends.mps.is_available():
            self.device_combo.addItem("MPS (Apple GPU)", "mps")
            self.device_combo.setCurrentIndex(1)  # Default to MPS if available
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA (Nvidia GPU)", "cuda")
            self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.on_load_model)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo, 1)
        device_layout.addWidget(self.load_model_btn)
        layout.addLayout(device_layout)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        lang_label.setMinimumWidth(80)
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Auto Detect", "auto")
        self.lang_combo.addItem("Japanese", "ja")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Chinese", "zh")
        self.lang_combo.addItem("Korean", "ko")
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo, 1)
        layout.addLayout(lang_layout)

        # Real-time checkbox
        realtime_layout = QHBoxLayout()
        self.realtime_checkbox = QCheckBox("Real-time Transcription")
        self.realtime_checkbox.setToolTip(
            "Transcribe audio in real-time as you speak.\n"
            "Works best with 'tiny' or 'base' models."
        )
        realtime_layout.addWidget(self.realtime_checkbox)
        realtime_layout.addStretch()
        layout.addLayout(realtime_layout)

        # Output folder (optional)
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Output:")
        folder_label.setMinimumWidth(80)
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("(Optional) Select folder to save transcriptions...")
        self.folder_btn = QPushButton("Browse")
        self.folder_btn.clicked.connect(self.on_browse_folder)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_input, 1)
        folder_layout.addWidget(self.folder_btn)
        layout.addLayout(folder_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Please load a model first.")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Recording buttons
        btn_layout = QHBoxLayout()
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.setEnabled(False)
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.record_btn.clicked.connect(self.on_toggle_recording)

        self.transcribe_btn = QPushButton("📝 Transcribe")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.setMinimumHeight(50)
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #2196F3;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.transcribe_btn.clicked.connect(self.on_transcribe)

        btn_layout.addWidget(self.record_btn)
        btn_layout.addWidget(self.transcribe_btn)
        layout.addLayout(btn_layout)

        # Result text
        result_label = QLabel("Transcription Result:")
        result_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Transcription will appear here...")
        self.result_text.setMinimumHeight(150)
        layout.addWidget(self.result_text)

        # Copy button
        copy_layout = QHBoxLayout()
        copy_layout.addStretch()
        self.copy_btn = QPushButton("📋 Copy to Clipboard")
        self.copy_btn.clicked.connect(self.on_copy)
        copy_layout.addWidget(self.copy_btn)
        layout.addLayout(copy_layout)

        # Apply modern clean theme
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: #eaf0fb;
                font-size: 14px;
            }
            QCheckBox {
                color: #eaf0fb;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #533483;
                border-radius: 4px;
                background-color: #0f3460;
            }
            QCheckBox::indicator:checked {
                background-color: #00d9ff;
                border-color: #00d9ff;
            }
            QComboBox {
                background-color: #0f3460;
                color: #eaf0fb;
                border: 2px solid #00d9ff;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #00f5d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #0f3460;
                color: #eaf0fb;
                selection-background-color: #00d9ff;
            }
            QLineEdit {
                background-color: #0f3460;
                color: #eaf0fb;
                border: 2px solid #533483;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #00d9ff;
            }
            QTextEdit {
                background-color: #0f3460;
                color: #eaf0fb;
                border: 2px solid #533483;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                line-height: 1.5;
            }
            QPushButton {
                background-color: #533483;
                color: #eaf0fb;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00d9ff;
                color: #1a1a2e;
            }
            QPushButton:pressed {
                background-color: #00b8d4;
                color: #1a1a2e;
            }
            QPushButton:disabled {
                background-color: #2a2a4a;
                color: #666688;
            }
            QProgressBar {
                border: 2px solid #533483;
                border-radius: 6px;
                text-align: center;
                background-color: #0f3460;
                color: #eaf0fb;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d9ff, stop:1 #00f5d4);
                border-radius: 4px;
            }
        """)

    def on_copy(self):
        """Copy transcription to clipboard."""
        text = self.result_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("✅ Copied to clipboard!")

    def on_browse_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_input.setText(folder)

    def on_load_model(self):
        """Load the selected Whisper model."""
        model_name = self.model_combo.currentText()
        device = self.device_combo.currentData()

        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Loading '{model_name}' on {device.upper()}...")
        self.load_model_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.device_combo.setEnabled(False)

        self.loader_thread = ModelLoaderThread(model_name, device)
        self.loader_thread.progress.connect(self.on_model_progress)
        self.loader_thread.finished_loading.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()

    def on_model_progress(self, message: str):
        """Update status with model loading progress."""
        self.status_label.setText(message)

    def on_model_loaded(self, model):
        """Handle model loaded successfully."""
        self.model = model
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ Model loaded! Ready to record.")
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.record_btn.setEnabled(True)

    def on_model_error(self, error: str):
        """Handle model loading error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Error: {error}")
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to load model:\n{error}")

    def on_toggle_recording(self):
        """Toggle recording on/off."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.realtime_transcription = ""
        self.result_text.clear()

        is_realtime = self.realtime_checkbox.isChecked()
        self.recorder.start(realtime_callback=is_realtime)

        self.record_btn.setText("⏹️ Stop Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #ff9800;
                color: #1a1a2e;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #ffc107; }
        """)

        if is_realtime:
            self.status_label.setText("🎤 Recording with real-time transcription...")
            self.transcribe_btn.setEnabled(False)
            # Start real-time processing timer (every 3 seconds)
            self.realtime_timer = QTimer()
            self.realtime_timer.timeout.connect(self.process_realtime_chunk)
            self.realtime_timer.start(3000)
        else:
            self.status_label.setText("🎤 Recording... Click Stop when done.")
            self.transcribe_btn.setEnabled(False)

    def process_realtime_chunk(self):
        """Process accumulated audio for real-time transcription."""
        if not self.is_recording or self.pending_transcription:
            return

        chunk = self.recorder.get_realtime_chunk()
        if chunk is not None and len(chunk) > self.recorder.sample_rate:  # At least 1 second
            self.pending_transcription = True
            language = self.lang_combo.currentData()

            self.rt_thread = RealtimeTranscribeThread(
                self.model, chunk, self.recorder.sample_rate, language
            )
            self.rt_thread.transcription_update.connect(self.on_realtime_update)
            self.rt_thread.error.connect(self.on_realtime_error)
            self.rt_thread.finished.connect(lambda: setattr(self, 'pending_transcription', False))
            self.rt_thread.start()

    def on_realtime_update(self, text: str):
        """Update with real-time transcription result."""
        self.realtime_transcription += text + " "
        self.result_text.setText(self.realtime_transcription.strip())
        # Scroll to bottom
        scrollbar = self.result_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_realtime_error(self, error: str):
        """Handle real-time transcription error."""
        print(f"Real-time error: {error}")

    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False

        # Stop real-time timer
        if self.realtime_timer:
            self.realtime_timer.stop()
            self.realtime_timer = None

        audio_data = self.recorder.stop()

        if len(audio_data) > 0:
            self.recorder.save(self.temp_audio_path, audio_data)
            duration = len(audio_data) / self.recorder.sample_rate

            if self.realtime_checkbox.isChecked():
                # Process any remaining audio
                self.status_label.setText(f"✅ Recorded {duration:.1f}s. Real-time transcription complete.")
                self.save_if_output_set()
            else:
                self.status_label.setText(f"✅ Recorded {duration:.1f}s. Ready to transcribe.")
                self.transcribe_btn.setEnabled(True)
        else:
            self.status_label.setText("⚠️ No audio recorded.")

        self.record_btn.setText("🔴 Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

    def save_if_output_set(self):
        """Save transcription if output folder is set."""
        output_folder = self.folder_input.text().strip()
        text = self.result_text.toPlainText().strip()

        if output_folder and os.path.isdir(output_folder) and text:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(output_folder) / f"transcription_{timestamp}.txt"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_label.setText(f"✅ Saved to: {filepath}")

    def on_transcribe(self):
        """Start transcription."""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        if not os.path.exists(self.temp_audio_path):
            QMessageBox.warning(self, "Warning", "No audio recording found.")
            return

        language = self.lang_combo.currentData()

        self.progress_bar.setVisible(True)
        self.transcribe_btn.setEnabled(False)
        self.record_btn.setEnabled(False)

        self.transcribe_thread = TranscribeThread(
            self.model, self.temp_audio_path, language
        )
        self.transcribe_thread.progress.connect(self.on_transcribe_progress)
        self.transcribe_thread.finished_transcribe.connect(self.on_transcribe_finished)
        self.transcribe_thread.error.connect(self.on_transcribe_error)
        self.transcribe_thread.start()

    def on_transcribe_progress(self, message: str):
        """Update transcription progress."""
        self.status_label.setText(message)

    def on_transcribe_finished(self, text: str):
        """Handle transcription complete."""
        self.progress_bar.setVisible(False)
        self.result_text.setText(text)
        self.transcribe_btn.setEnabled(True)
        self.record_btn.setEnabled(True)

        # Save to file if output folder specified
        self.save_if_output_set()
        if not self.folder_input.text().strip():
            self.status_label.setText("✅ Transcription complete!")

    def on_transcribe_error(self, error: str):
        """Handle transcription error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Error: {error}")
        self.transcribe_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Transcription failed:\n{error}")


def main():
    """Entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Whisper Transcription")

    window = WhisperApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
