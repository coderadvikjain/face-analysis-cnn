# Face Gender & Age Prediction using CNN (PyTorch)

A deep learning project that predicts **gender** (classification) and **age** (regression) from face images using custom CNNs built with PyTorch. Developed and trained on Kaggle.

---

## Overview

This project builds two separate CNN models:

- **GenderCNN** — classifies gender using `CrossEntropyLoss`
- **AgeCNN** — predicts age as a continuous value using `MSELoss`

Both models share a similar 3-layer convolutional architecture with fully connected heads, trained on the Kaggle face dataset.

---

## Dataset

- **Source:** [Kaggle — Face Dataset (SEP-25 DL Gen AI NPPE)](https://www.kaggle.com/competitions/sep-25-dl-gen-ai-nppe-1)
- **Structure:** `train.csv` and `test.csv` with image paths, gender labels, and age values
- **Split:** 80/20 train-validation split (`random_state=42`)

---

## Model Architecture

| Layer         | GenderCNN              | AgeCNN                 |
|---------------|------------------------|------------------------|
| Conv Block 1  | Conv2d(3→32) + ReLU + MaxPool | Conv2d(3→32) + ReLU + MaxPool |
| Conv Block 2  | Conv2d(32→64) + ReLU + MaxPool | Conv2d(32→64) + ReLU + MaxPool |
| Conv Block 3  | Conv2d(64→128) + ReLU + MaxPool | Conv2d(64→128) + ReLU + MaxPool |
| FC Head       | Flatten → Dense → Softmax (2 classes) | Flatten → Dense → Linear (1 output) |

---

## Training Details

| Parameter       | GenderCNN         | AgeCNN            |
|-----------------|-------------------|-------------------|
| Optimizer       | Adam              | Adam              |
| Learning Rate   | 0.001             | 0.0005            |
| Epochs          | 10                | 10                |
| Batch Size      | 64                | 64                |
| Loss Function   | CrossEntropyLoss  | MSELoss           |

**Data Augmentation:** RandomHorizontalFlip, RandomRotation(10°), ColorJitter (brightness, contrast, saturation), Resize to 224×224.

---

## Results

| Approach | Accuracy |
|----------|----------|
| Pretrained models (best on leaderboard) | **0.85** |
| **This project (CNN from scratch)** | **0.75** |

> Achieved **0.75 accuracy with a fully custom CNN built from scratch** — no transfer learning, no pretrained weights. The top leaderboard score using pretrained models is 0.85, putting this scratch-built approach within striking distance despite the simplicity of the architecture.

---

## Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook
├── Report.pdf              # Report on Notebook
└── README.md               # This file
```

---

## Pretrained Weights

Model weights are hosted on Kaggle Model Hub (too large for GitHub):

**Download:** [jatinnahatajain/gender-age-cnn](https://www.kaggle.com/models/jatinnahatajain/gender-age-cnn)

| Model | File | Size |
|-------|------|------|
| GenderCNN | `gender_best.pth` | ~50 MB |
| AgeCNN | `age_best.pth` | ~199 MB |

To load in code:
```python
import kagglehub
kagglehub.model_download("jatinnahatajain/gender-age-cnn/pytorch/baseline")
```

---

## How to Run

1. **On Kaggle / Google Colab (recommended):**
   - Open the [notebook on Kaggle](https://www.kaggle.com/) and click **"Copy & Edit"**
   - Or upload `notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
   - Enable **GPU accelerator** in notebook settings
   - Run all cells

2. **Locally:**
   ```bash
   pip install torch torchvision pandas numpy pillow kagglehub
   jupyter notebook notebook.ipynb
   ```

---

## Tech Stack

- Python 3
- PyTorch
- Torchvision
- Pandas / NumPy
- PIL (Pillow)

---

## Key Features

- Custom CNN architectures (no pre-trained backbones)
- Separate specialized models for classification vs regression tasks
- Image augmentation pipeline for better generalization
- Model saving & loading via Kaggle Model Hub
- GPU-accelerated training

---

## Author

**Jatin Nahata Jain** — [Kaggle Profile](https://www.kaggle.com/jatinnahatajain)

---

## License

This project is open source and available under the [MIT License](LICENSE).