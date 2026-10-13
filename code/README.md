# Dual-Stream Progressive Hazard Network (DSPHN)
## AIN3002 Final Project, 2026
### Mehreen Faisal, Department of Artificial Intelligence Engineering

---

## Overview

DSPHN is a novel semantic segmentation architecture for indoor hazard detection from a single RGB image. It integrates two paradigms:

1. **DAHD dual-stream decoding**: Appearance Decoder (standard conv) + Context Decoder (dilated conv) with Severity Fusion Gates
2. **PHRN hierarchical inference**: Stage 1 binary hazard map conditions Stage 2 four-class segmentation

**4 Classes**: Background | Safe Floor | Obstacle | Sharp Objects

---

## Results

| Class | IoU | Dice |
|---|---|---|
| Background | 0.8081 | 0.8938 |
| Safe Floor | 0.5517 | 0.7111 |
| Obstacle | 0.3195 | 0.4843 |
| Sharp Objects | 0.0000 | 0.0000 |
| **Mean** | **0.4198** | **0.5223** |

**Best Val mIoU: 0.4847 | Pixel Accuracy: 0.8332**

---

## Ablation Study

| Model | mIoU |
|---|---|
| Baseline U-Net | 0.5771 |
| DAHD only | 0.3894 |
| PHRN only | 0.3793 |
| **DSPHN (proposed)** | **0.4198** |

---

## Dataset

- ADE20K: 3,000 images (base indoor scenes)
- Manual annotations: 100 images, Make Sense AI (hazard-specific labels)
- COCO: 186 knife/scissors images (Sharp Objects)
- Split: 70/15/15 train/val/test, seed=42

---

## Usage

```python
# Run DSPHN_Final_Notebook.ipynb in Google Colab (T4 GPU)
# Requires: torch, torchvision, pycocotools, numpy, PIL, sklearn

model = DualStreamProgressiveHazardNetwork(num_classes=4).to(device)
model.load_state_dict(torch.load('best_DSPHN_model.pth'))
model.eval()

with torch.no_grad():
    stage2_seg, stage1_map = model(image_tensor)
    prediction = torch.argmax(stage2_seg, dim=1)
    hazard_alert = stage1_map  # binary hazard probability map
```

---

## File Structure

```
code/
├── DSPHN_Final_Notebook.ipynb  # Complete training and evaluation
└── README.md
report/
└── DSPHN_Report.pdf            # 13-page CVPR format report
poster/
└── DSPHN_Poster.html           # Academic poster (open in browser, print to PDF)
```

---

## Architecture

```
Input RGB (3x256x256)
      |
 Shared Encoder (64->128->256->512->1024)
      |
 +----+----+
 |         |
App      Context
Decoder  Decoder
(std)    (dilated)
 |         |
 +--Fusion--+
    Gate -> f4
      |
   Stage 1 Head -> Binary Map M1
      |
   Stage 2 Head([f4; M1]) -> 4-class output
```

---

## Citation

```bibtex
@article{dsphn2026,
  title={DSPHN: Dual-Stream Progressive Hazard Network for Indoor Scene Segmentation},
  author={Mehreen Faisal},
  journal={AIN3002 Final Project},
  year={2026}
}
```
