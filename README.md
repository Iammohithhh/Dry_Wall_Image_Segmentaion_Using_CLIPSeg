# Prompted Segmentation for Drywall Quality Assurance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

A text-conditioned segmentation system that produces binary masks from natural language prompts like `"segment crack"` and `"segment taping area"`.

##  Results

| Prompt | Test Samples | mIoU | Dice | Precision | Recall |
|--------|--------------|------|------|-----------|--------|
| `segment crack` | 804 | **0.4723** | 0.6180 | 0.6274 | 0.7101 |
| `segment taping area` | 153 | **0.6069** | 0.7457 | 0.7475 | 0.7849 |
| **Overall** | **957** | **0.4938** | **0.6384** | — | — |

**Inference Speed:** 45.4 ± 0.9 ms/image (NVIDIA T4)

##  Architecture

We fine-tune [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) with frozen CLIP encoders and trainable decoder:

```
Image  ──► CLIP Vision Encoder (frozen) ──┐
                                          ├──► Decoder (trained) ──► Binary Mask
Prompt ──► CLIP Text Encoder (frozen)  ───┘
```

**Why CLIPSeg?**
- Native text conditioning (no complex pipelines)
- Efficient fine-tuning (only 1.2% parameters trained)
- Production-ready inference (45ms/image)

##  Repository Structure

```
├── drywall_segmentation_clipseg.ipynb  # Main training notebook
├── report/
│   └── drywall_segmentation_report.pdf # Full report with analysis
├── outputs/
│   ├── predictions/
│   │   ├── cracks/                     # Crack prediction masks
│   │   └── taping/                     # Taping prediction masks
│   ├── training_curves.png
│   ├── prediction_visualization.png
│   └── prompt_variants.png
├── checkpoints/
│   └── best_model.pt                   # Trained weights
└── README.md
```

##  Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/drywall-segmentation/blob/main/drywall_segmentation_clipseg.ipynb)

### 2. Or run locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/drywall-segmentation.git
cd drywall-segmentation

# Install dependencies
pip install torch torchvision transformers albumentations opencv-python roboflow

# Run inference
python inference.py --image path/to/image.jpg --prompt "segment crack"
```

### 3. Inference Example

```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from PIL import Image

# Load model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load fine-tuned weights
checkpoint = torch.load("checkpoints/best_model.pt", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
image = Image.open("test_image.jpg")
inputs = processor(text=["segment crack"], images=[image], return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    mask = torch.sigmoid(outputs.logits) > 0.5
```

##  Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `CIDAS/clipseg-rd64-refined` |
| Image Size | 352 × 352 |
| Batch Size | 8 (×2 gradient accumulation) |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (weight decay = 0.01) |
| Scheduler | OneCycleLR |
| Epochs | 25 |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Trainable Params | 1.78M / 150.7M (1.2%) |

##  Datasets

| Dataset | Source | Train | Val | Test | Format |
|---------|--------|-------|-----|------|--------|
| Cracks | [Roboflow](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | 3,758 | 805 | 806 | COCO (polygons) |
| Taping | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | 715 | 153 | 154 | YOLO (bboxes) |

**Note:** The original Cracks dataset was Object Detection format. We converted it to Instance Segmentation (see report Section 3 for details).

##  Prompt Consistency

The model responds consistently to semantic variations of the same prompt—all variants produce nearly identical predictions, confirming effective text conditioning.

##  Known Limitations

| Limitation | Cause | Mitigation |
|------------|-------|------------|
| Hairline cracks (<5px) missed | 64×64 decoder resolution bottleneck | Multi-scale inference or SAM decoder |
| Outdoor surfaces (brick, concrete) | Domain shift from indoor training data | Data augmentation |
| Low contrast cracks | CLIP emphasizes semantics over gradients | Edge-enhanced inputs |

##  Recommended Improvements

| Approach | Key Idea | Expected Gain | Trade-off |
|----------|----------|---------------|-----------|
| **CLIP + SAM Decoder** | Use CLIP text encoder + SAM's high-res decoder | +15-20% boundary IoU | 3× slower |
| **Multi-scale Fusion** | Run at 3 scales, aggregate predictions | +5-10% on thin structures | 3× compute |
| **CRF Post-processing** | Refine boundaries using image edges | +3-5% boundary precision | +20ms/image |

##  Reproducibility

- **Random Seed:** 42
- **Framework:** PyTorch 2.x, HuggingFace Transformers
- **Hardware:** NVIDIA Tesla T4 (Google Colab)
- **Training Time:** 1 hour 4 minutes



## Author

Mohith | IIT Bombay | March 2026
