# Whisper Mac Record

A native macOS desktop application for audio transcription using OpenAI Whisper.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- 🎤 **Audio Recording**: Record directly from your microphone with start/stop buttons
- 🤖 **Model Selection**: Choose from Whisper models (tiny, base, small, medium, large)
- 📥 **Auto Download**: Models are automatically downloaded on first use with progress indicator
- 🌐 **Language Support**: Auto-detect or specify language (Japanese, English, Chinese, Korean)
- 💾 **Save Transcriptions**: Automatically save results to a specified folder with timestamps

## Prerequisites

1. **macOS** with Python 3.10 or later
2. **ffmpeg** (required for audio processing):

```bash
brew install ffmpeg
```

3. **uv** (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/whisper-mac-record.git
cd whisper-mac-record

# Install dependencies
uv sync
```

## Usage

```bash
# Run the application
uv run python src/main.py
```

### Quick Start

1. **Load Model**: Select a model from the dropdown and click "Load Model"
   - First-time loading will download the model (this may take a few minutes)
2. **Record Audio**: Click "Start Recording" and speak into your microphone
3. **Stop Recording**: Click "Stop Recording" when finished
4. **Transcribe**: Click "Transcribe" to convert speech to text
5. **Save** (Optional): Set an output folder to automatically save transcriptions

## Model Comparison

| Model  | Parameters | Size   | Speed   | Accuracy |
|--------|-----------|--------|---------|----------|
| tiny   | 39M       | ~75MB  | Fastest | Basic    |
| base   | 74M       | ~140MB | Fast    | Good     |
| small  | 244M      | ~460MB | Medium  | Better   |
| medium | 769M      | ~1.5GB | Slow    | Great    |
| large  | 1550M     | ~3GB   | Slowest | Best     |

### Microphone Permission

On first run, macOS will prompt for microphone access. Grant permission in:
**System Preferences > Security & Privacy > Privacy > Microphone**

### Model Download Issues

If model download fails, check your internet connection and try again. Models are cached in `~/.cache/whisper/`.

## License

MIT License

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - Desktop UI framework
