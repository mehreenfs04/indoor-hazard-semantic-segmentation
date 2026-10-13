# indoor-hazard-semantic-segmentation
A computer vision project exploring semantic segmentation for indoor scene understanding and hazard-oriented environmental analysis.
# Dual-Stream Progressive Hazard Network (DSPHN): Indoor Hazard Segmentation

DSPHN is a semantic segmentation architecture that identifies safety hazards in indoor scenes from a single RGB image. Instead of labeling objects by what they are, it labels pixels by the injury risk they pose:

| Class | Injury risk |
|---|---|
| Background | None |
| Safe Floor | Slip/trip |
| Obstacle | Collision |
| Sharp Objects | Laceration |

Built for the Deep Learning for Image Processing course (AIN3002).

## Architecture

DSPHN has three key components:

1. **Dual-Stream Appearance-Context Decoder:** Two parallel decoders, one using standard convolutions for local texture cues and one using dilated convolutions for spatial context.
2. **Severity-Guided Adaptive Fusion Gates:** Learnable gates, initialised from hazard severity scores, that blend the two streams at each decoder level.
3. **Two-Stage Hazard Refinement:** Stage 1 predicts a binary safe/unsafe map, which feeds directly into Stage 2 for fine-grained four-class classification.

## Data

I used 3,000 ADE20K indoor images remapped to the hazard taxonomy, plus 100 manually annotated images (Make Sense AI) and 186 COCO knife/scissors images to boost the rare Sharp Objects class. The data is split 70/15/15 (2,100 train / 450 validation / 450 test).

## Results

On the held-out test set, DSPHN reached a **mean IoU of 0.4198** and **pixel accuracy of 0.8332**.

| Model | Dual-stream + gates | Two-stage | mIoU |
|---|:---:|:---:|:---:|
| Baseline U-Net | ✗ | ✗ | 0.5771* |
| Dual-stream + gates only | ✓ | ✗ | 0.3894 |
| Two-stage only | ✗ | ✓ | 0.3793 |
| **DSPHN (full)** | ✓ | ✓ | **0.4198** |

\*The baseline was trained without the Sharp Objects class, so it is an easier task and not directly comparable.

The ablation shows the full model outperforms both the dual-stream-only and two-stage-only variants, confirming that the two designs each contribute.

| Class | IoU |
|---|---|
| Background | 0.8081 |
| Safe Floor | 0.5517 |
| Obstacle | 0.3195 |
| Sharp Objects | 0.0000 |

## Limitations and Future Work

Sharp Objects scored zero IoU due to extreme pixel sparsity (about 0.06% of test pixels), despite class weighting and COCO augmentation. Training was also limited to 15 epochs at 256×256 resolution on a T4 GPU.

Future work includes larger balanced hazard datasets, better rare-class balancing (e.g. patch-level oversampling), and a lightweight encoder for real-time edge deployment.

## Built With

Python, PyTorch, torchvision, NumPy, scikit-learn, Pillow, Matplotlib, pycocotools, Google Colab (NVIDIA T4 GPU)
