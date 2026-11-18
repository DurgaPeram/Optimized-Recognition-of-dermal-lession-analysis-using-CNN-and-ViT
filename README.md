# DermaVision

**AI-assisted dermal lesion recognition using an EfficientNet-V2-S and Swin Transformer ensemble**

DermaVision is an educational full-stack application for analyzing dermatoscopic images and estimating the probability of seven skin-lesion classes from the HAM10000 dataset. The prediction service combines an EfficientNet-V2-S CNN and a Swin Tiny Vision Transformer, while the responsive frontend presents the predicted class, confidence, risk level, and disease-specific educational information.

> [!WARNING]
> **Medical disclaimer:** DermaVision is for educational and research purposes only. It is not a medical device, does not provide a clinical diagnosis, and must not be used as a substitute for professional medical advice, examination, or treatment. Uploaded images should not contain personal identifiers.

## Table of Contents

- [Key Features](#key-features)
- [Supported Lesion Classes](#supported-lesion-classes)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Dataset and Checkpoints](#dataset-and-checkpoints)
- [Installation](#installation)
- [Model Training](#model-training)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- Drag-and-drop or file-selector image upload with a local preview
- Ensemble inference from EfficientNet-V2-S and Swin Tiny
- Predicted lesion name, confidence percentage, and educational risk level
- Disease-specific causes, symptoms, treatments, prevention, and recovery information
- Responsive pages for analysis, disease summaries, details, project information, and contact
- FastAPI prediction service with interactive Swagger documentation
- HAM10000 preparation, training, checkpoint, and ensemble-evaluation scripts
- CUDA and mixed-precision support when compatible hardware is available

## Supported Lesion Classes

| Dataset label | Display name |
|---|---|
| `akiec` | Actinic Keratosis |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesion |
| `mel` | Melanoma |

## System Architecture

```text
Browser / Static Frontend
        │
        │  Multipart POST: image + HTTP request
        ▼
FastAPI Prediction Server
        │
        ├── EfficientNet-V2-S preprocessing and inference
        │       384 × 384 input
        │
        └── Swin Tiny preprocessing and inference
                224 × 224 input
        │
        ▼
Average model logits
        │
        ▼
Softmax probabilities
        │
        ▼
Predicted class + confidence
```

The backend loads both trained checkpoints, prepares one image for each model's required input size and normalization pipeline, averages their logits, and applies softmax to produce class probabilities.

## Technology Stack

| Area | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Deep learning | PyTorch, TorchVision, TIMM |
| Vision preprocessing | Albumentations, Pillow, NumPy |
| Dataset utilities | pandas, scikit-learn |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Dataset | HAM10000 |

## Project Structure

```text
.
├── backend/
│   └── app.py                 # FastAPI prediction server
├── frontend/
│   ├── index.html             # Landing page
│   ├── analyze.html           # Image upload interface
│   ├── diseases.html          # Lesion overview page
│   ├── disease-details.html   # Prediction information page
│   ├── about.html             # Project information
│   ├── contact.html           # Contact page
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── api.js             # Upload and API integration
│       ├── app.js             # Prediction details rendering
│       └── diseases-data.js   # Educational lesion data
├── training/
│   ├── dataset.py             # HAM10000 dataset and transforms
│   ├── split.py               # Stratified train/validation split
│   ├── class_weights.py       # Class-imbalance weights
│   ├── losses.py              # Focal-loss implementation
│   ├── train_effnet.py        # EfficientNet-V2-S training
│   ├── train_swin.py          # Swin Transformer training
│   ├── train.py               # EfficientNet smoke test
│   └── ensemble_eval.py       # Ensemble metrics and reports
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.9 or newer
- `pip` and a virtual-environment tool
- A modern web browser
- For training: a CUDA-capable NVIDIA GPU is recommended
- For CPU-only use: inference is possible, but significantly slower

A GPU with sufficient memory is recommended for training. Exact memory requirements depend on image size, batch size, mixed-precision settings, and augmentation pipelines.

## Dataset and Checkpoints

### HAM10000 Dataset

Download the HAM10000 dataset and metadata from the official Kaggle or Harvard Dataverse source, then organize them locally as follows:

```text
data/
└── raw/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/

models/
├── effnet.pt
└── swin.pt
```

The `data/` and `models/` directories are excluded from version control through `.gitignore` because of their size and licensing/data-usage requirements. The raw images are not included in this repository.

### Model Checkpoints

Both checkpoint files must exist before starting the prediction server:

- `models/effnet.pt` — trained EfficientNet-V2-S weights
- `models/swin.pt` — trained Swin Tiny weights

Pretrained checkpoints are not included. Train them locally using the instructions below or provide your own compatible checkpoints in the same locations.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DurgaPeram/Optimized-Recognition-of-dermal-lession-analysis-using-CNN-and-ViT.git
cd Optimized-Recognition-of-dermal-lession-analysis-using-CNN-and-ViT
```

### 2. Create a virtual environment

**Linux or macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn torch torchvision timm albumentations pillow numpy pandas scikit-learn python-multipart
```

For a GPU build of PyTorch, install the CUDA-enabled wheel that matches the installed NVIDIA driver and CUDA toolkit. Use the official PyTorch installation selector rather than assuming the CPU/Python packages include GPU support.

## Model Training

Before training, ensure the HAM10000 metadata and image folders are available under `data/raw/`.

### 1. Create the dataset split and class weights

```bash
cd training
python split.py
python class_weights.py
```

The split script uses stratification to preserve class distributions across the training and validation sets.

### 2. Train EfficientNet-V2-S

```bash
python train_effnet.py
```

The EfficientNet pipeline uses focal loss, label smoothing, class weights, staged backbone freezing/fine-tuning, mixed precision, and the configured image augmentations.

### 3. Train Swin Tiny

```bash
python train_swin.py
```

The Swin pipeline uses weighted cross-entropy, AdamW, staged training, and mixed precision.

### 4. Evaluate the ensemble

```bash
python ensemble_eval.py
```

Training scripts automatically use CUDA when it is detected. The best checkpoints are intended to be saved as:

```text
models/effnet.pt
models/swin.pt
```

## Running the Application

### Start the FastAPI backend

From the repository root:

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the frontend

Open a second terminal from the repository root:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

The frontend sends requests to:

```text
http://127.0.0.1:8000/predict
```

If the backend host or port changes, update the API URL in `frontend/js/api.js`.

## API Reference

### Health check

```http
GET /
```

### Predict a lesion

```http
POST /predict
Content-Type: multipart/form-data
```

**Form fields:**

| Field | Type | Required | Description |
|---|---|---:|---|
| `file` | image file | Yes | Dermatoscopic image upload |

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@skin-lesion.jpg"
```

**Example success response:**

```json
{
  "prediction": "Melanoma",
  "confidence": 94.73
}
```

The exact field set may vary if the backend implementation is updated. Treat the response as an application contract and update this section whenever the API changes.

## Configuration

- Backend host and port: configure in the `uvicorn` command or deployment configuration
- Frontend API URL: `frontend/js/api.js`
- EfficientNet checkpoint: `models/effnet.pt`
- Swin checkpoint: `models/swin.pt`
- Dataset metadata: `data/raw/HAM10000_metadata.csv`
- Dataset images: `data/raw/HAM10000_images_part_1/` and `HAM10000_images_part_2/`

CORS settings in `backend/app.py` may need to be restricted or expanded before public deployment.

## Evaluation

The ensemble evaluation script reports:

- Overall accuracy
- Macro F1 score
- Per-class precision, recall, and F1 score
- Classification report

The repository does not currently publish measured metrics or model checkpoints. Record the dataset split, random seed, software versions, hardware, training settings, and measured results here before treating model performance as reproducible or production-ready.

## Limitations

- Pretrained checkpoints are not distributed with this repository
- HAM10000 must be obtained and prepared separately
- HAM10000 is a benchmark dataset and may not represent every skin tone, lesion type, or clinical environment
- Ensemble confidence is not a calibrated probability of disease presence
- The static frontend has no authentication or server-side persistent storage
- Uploaded images are handled by the local development server and should not be used with personally identifiable or sensitive medical data
- Development settings, CORS configuration, and local checkpoint loading are not hardened for public deployment
- Predictions should not be interpreted as clinical diagnoses

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a branch from `main`.
3. Make focused code or documentation changes.
4. Test the backend and frontend locally.
5. Commit the changes with a clear message.
6. Open a pull request describing the motivation, implementation, and test steps.

Please do not commit datasets, private medical images, or trained model checkpoints unless they are intentionally licensed and safe to distribute.

## License

No license file is currently included. A license is required before the project can be confidently reused or redistributed. Until one is added, all rights remain with the project owner.

When a license is selected, replace this section with the license name and a link to `LICENSE`, for example:

```text
MIT License

See LICENSE for the full license text.
```
