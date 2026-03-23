# VisionScript AI

**AI-powered image captioning using ResNet-50 + GPT-2**

VisionScript is an end-to-end image captioning system that encodes images with a pretrained ResNet-50 and decodes natural language captions with GPT-2. It includes a full-stack web interface for interactive use.

---

## Demo

Upload any image and VisionScript generates a natural language caption in real time.

> *"A man riding a bicycle down a street next to a building"*

---

## Architecture

```
Image → ResNet-50 Encoder → Linear Projection (2048→768) → Visual Token
                                                                  ↓
                                                GPT-2 Decoder (prepend visual token)
                                                                  ↓
                                                         Generated Caption
```

### Encoder — `ResNetEncoder`
- Pretrained **ResNet-50** (ImageNet) with the classification head removed
- Extracts a `[batch, 2048]` feature vector via global average pooling
- A learned **linear projection** maps `2048 → 768` to match GPT-2's embedding dimension
- Output shape: `[batch, 1, 768]` — a single "visual token"

### Decoder — `GPT2Decoder`
- Pretrained **GPT-2 small** (117M parameters)
- The visual token is **prepended** to the token embeddings at each forward pass
- GPT-2 generates captions autoregressively conditioned on the image

### Training Strategy (Two-Phase)
| Phase | Epochs | Encoder | Learning Rate |
|---|---|---|---|
| Phase 1 | 1–3 | Frozen | `3e-4` |
| Phase 2 | 4–5 | Unfrozen | `3e-5` |

Freezing the encoder in Phase 1 trains the projection layer and GPT-2 efficiently before full end-to-end fine-tuning in Phase 2.

---

## Dataset

Trained on **MS COCO 2017** — 118,000 training images with 5 human-written captions each (~591,000 caption pairs total).

---

## Project Structure

```
Image_Captioning_CV_Project/
├── src/
│   ├── model.py        # VisionScript: top-level model combining encoder + decoder
│   ├── encoder.py      # ResNetEncoder: ResNet-50 → visual token
│   ├── decoder.py      # GPT2Decoder: GPT-2 conditioned on visual token
│   ├── dataset.py      # COCOCaptionDataset: COCO data loader + preprocessing
│   ├── train.py        # Two-phase training loop
│   ├── inference.py    # load_model() and caption_image() utilities
│   └── evaluate.py     # Evaluation utilities
├── web/
│   ├── app.py          # Flask REST API (port 5001)
│   └── frontend/       # React + Vite + Tailwind frontend
│       └── src/
│           └── App.jsx # Drag-and-drop UI with live captioning
├── models/
│   └── visionscript_final.pt  # Trained model weights (download separately — see README)
└── data/
```

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- A CUDA GPU, Apple MPS, or CPU

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Image_Captioning_CV_Project.git
cd Image_Captioning_CV_Project
```

### 2. Install Python dependencies

```bash
pip install torch torchvision transformers flask flask-cors pillow tqdm
```

### 3. Install frontend dependencies

```bash
cd web/frontend
npm install
```

---

## Model Weights

The trained model weights are too large to store in this repository (~500 MB). Download them from Google Drive and place the file at `models/visionscript_final.pt`.

**[⬇️ Download visionscript_final.pt](https://drive.google.com/file/d/1Dlx_BcNqqbeO_jS28My4zmhwyKNSTv5m/view?usp=share_link)**

```bash
# macOS / Linux — After downloading, move the file into the models/ directory
mv ~/Downloads/visionscript_final.pt models/visionscript_final.pt
```

```powershell
# Windows (PowerShell) — After downloading, move the file into the models/ directory
Move-Item "$env:USERPROFILE\Downloads\visionscript_final.pt" models\visionscript_final.pt
```

> The `models/` directory is listed in `.gitignore` — the weights will not be committed to the repo.

---

## Running the Web App

You'll need a trained model checkpoint placed at `models/visionscript_final.pt` (see [Model Weights](#model-weights) above).

**Terminal 1 — Flask backend:**
```bash
python3 web/app.py
# Runs on http://localhost:5001
```

**Terminal 2 — React frontend:**
```bash
cd web/frontend
npm run dev
# Runs on http://localhost:5173
```

Then open **http://localhost:5173** in your browser.

> **Note for macOS users:** Port 5000 is reserved by AirPlay Receiver on macOS Monterey+. The Flask backend runs on port **5001** to avoid this conflict.

---

## Training

Training was done on **Google Colab** with an A100 GPU using the MS COCO 2017 dataset.

```python
config = {
    'image_dir':       '/content/train2017',
    'annotation_file': '/content/annotations/captions_train2017.json',
    'checkpoint_dir':  '/content/drive/MyDrive/visionscript/models',
    'batch_size':      16,
    'learning_rate':   3e-4,
    'epochs':          5,
}
```

To train from scratch:
```bash
python3 -m src.train
```

Checkpoints are saved after each epoch as `checkpoint_epoch_{n}.pt`.

---

## API Reference

The Flask backend exposes two endpoints:

### `POST /caption`
Upload an image and receive a generated caption.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{ "caption": "a dog sitting on a couch next to a window" }
```

### `GET /health`
Check server status and active device.

**Response:**
```json
{ "status": "ok", "device": "mps" }
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Image Encoder | ResNet-50 (torchvision) |
| Language Decoder | GPT-2 small (HuggingFace Transformers) |
| Training | PyTorch, MS COCO 2017 |
| Backend API | Flask + Flask-CORS |
| Frontend | React 19, Vite, Tailwind CSS |
| Acceleration | CUDA / Apple MPS / CPU |

---

## Requirements

```
torch
torchvision
transformers
flask
flask-cors
pillow
tqdm
```
