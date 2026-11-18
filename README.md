# DermaVision — AI Dermal Lesion Analysis

DermaVision is a full-stack skin-disease classification system that combines EfficientNet-V2-S and a Swin Transformer to analyze dermatoscopic images. The application classifies skin lesions into seven HAM10000 categories and provides supporting disease, risk, and treatment information.

> ⚕️ **Medical disclaimer:** This project is for educational and research purposes only. It is not a substitute for professional diagnosis or medical advice.

## Features

- Upload and preview skin-lesion images.
- Ensemble inference using EfficientNet-V2-S and Swin Transformer.
- Confidence percentage and risk-level indication.
- Disease-specific causes, symptoms, treatments, prevention, and recovery details.
- Responsive HTML, CSS, and JavaScript frontend.
- FastAPI prediction service with CORS support.
- HAM10000 dataset preparation and model training scripts.

## Supported Conditions

| Dataset Label | Display Name |
|---|---|
| `akiec` | Actinic Keratosis |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `nv` | Melanocytic Nevus |
| `vasc` | Vascular Lesion |
| `mel` | Melanoma |

## Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Deep learning:** PyTorch, TorchVision, TIMM
- **Image processing:** Albumentations, Pillow, NumPy
- **Training utilities:** pandas, scikit-learn
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Dataset:** HAM10000

## Project Structure

```text
.
├── backend/
│   └── app.py                 # FastAPI inference server
├── frontend/
│   ├── index.html             # Landing page
│   ├── analyze.html           # Image upload and prediction interface
│   ├── diseases.html          # Disease overview page
│   ├── disease-details.html   # Detailed prediction information
│   ├── about.html             # Project information
│   ├── contact.html           # Contact page
│   ├── css/style.css          # Shared styling
│   └── js/
│       ├── api.js             # Image upload and API integration
│       ├── app.js             # Local prediction-details rendering
│       └── diseases-data.js   # Disease information
├── training/
│   ├── dataset.py             # HAM10000 dataset and transforms
│   ├── split.py               # Stratified train/validation split
│   ├── class_weights.py       # Class imbalance weight calculation
│   ├── losses.py              # Focal loss implementation
│   ├── train_effnet.py        # EfficientNet-V2-S training
│   ├── train_swin.py          # Swin Transformer training
│   ├── train.py               # One-batch EfficientNet test
│   └── ensemble_eval.py       # Ensemble evaluation and metrics
└── .gitignore
```

## Model Architecture

1. An uploaded image is resized and normalized independently for each model.
2. EfficientNet-V2-S processes the image at `384 × 384`.
3. Swin Tiny processes the image at `224 × 224`.
4. The logits from both models are averaged.
5. Softmax produces the final class probabilities.
6. The highest-probability class and confidence are returned through `/predict`.

## Dataset Setup

Download the HAM10000 dataset and metadata, then arrange them as follows:

```text
data/
├── raw/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/

models/
├── effnet.pt
└── swin.pt
```

The `data/` and `models/` directories are intentionally excluded through `.gitignore`.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/DurgaPeram/Optimized-Recognition-of-dermal-lession-analysis-using-CNN-and-ViT.git
cd Optimized-Recognition-of-dermal-lession-analysis-using-CNN-and-ViT
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn torch torchvision timm albumentations pillow numpy pandas scikit-learn python-multipart
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

## Training

Create a stratified dataset split:

```bash
cd training
python split.py
python class_weights.py
```

Train the CNN:

```bash
python train_effnet.py
```

Train the Vision Transformer:

```bash
python train_swin.py
```

Evaluate the ensemble:

```bash
python ensemble_eval.py
```

The training scripts use GPU acceleration automatically when CUDA is available. Best-performing checkpoints are saved under `../models/effnet.pt` and `../models/swin.pt`.

## Run the Application

Start the FastAPI server:

```bash
cd backend
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Serve the frontend from the repository root:

```bash
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

The frontend sends prediction requests to:

```text
http://127.0.0.1:8000/predict
```

## API Usage

### Prediction Endpoint

```http
POST /predict
Content-Type: multipart/form-data
```

Example with `curl`:

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict" \
  -F "file=@skin-lesion.jpg"
```

Example response:

```json
{
  "prediction": "Melanoma",
  "confidence": 94.73
}
```

## Training Details

- HAM10000 metadata is mapped to seven numerical classes.
- Training and validation images use separate Albumentations pipelines.
- Image augmentations include flipping, brightness/contrast changes, shifting, scaling, and rotation.
- The EfficientNet pipeline uses focal loss with label smoothing and class weights.
- The Swin pipeline uses weighted cross-entropy.
- Both models use AdamW and mixed-precision training.
- EfficientNet and Swin use staged backbone freezing and fine-tuning.
- Ensemble evaluation reports accuracy, macro F1 score, and a classification report.

## Limitations

- Pretrained model checkpoints are not included in the repository.
- The raw HAM10000 dataset is excluded due to repository size and must be obtained separately.
- The static frontend is designed for local use and contains no authentication or persistent server-side storage.
- Predictions and detailed medical guidance should not be interpreted as clinical diagnoses.

## Contributing

Contributions are welcome:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request with a clear description and test steps.

## License

No license file is currently included. A license should be added before the project is reused or redistributed.
