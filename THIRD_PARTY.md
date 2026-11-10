# Credits and model provenance

The detector was initialized from the COCO-pretrained YOLO11m segmentation model supplied by Ultralytics and fine-tuned on the competition training images. This repository distributes the fitted detector and loads it with Ultralytics 8.4.29. The upstream model carries the AGPL-3.0 license. [Model source](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) · [Upstream license](licenses/Ultralytics-AGPL-3.0.txt)

The seed-conditioned ROI refiner draws on XingJia He's detect-and-refine implementation. Its encoder is Torchvision's ResNet-18 and was initialized with the official ImageNet weights before filament training. The refiner distributed here was trained within this project. [Reference implementation](https://github.com/HeShen-1/filament-segmentation-2026/tree/b41dae215a87c6cb6dfbcf1459a4819021e0092d) · [MIT notice](licenses/HeShen-MIT.txt) · [Torchvision notice](licenses/Torchvision-BSD-3-Clause.txt) · [Encoder initialization](https://download.pytorch.org/models/resnet18-f37072fd.pth)

The spatial quality heads use a predicted instance mask together with its local image to estimate its contribution to panoptic quality. This design draws on Huang et al., *Mask Scoring R-CNN* (CVPR 2019). The four fitted heads and the ridge coefficients were produced for this entry. [Paper](https://arxiv.org/abs/1903.00241)

The notebook takes the organizers' GONG H-alpha JPEGs as input. Competition participants can obtain them from the official data page under the competition rules. The report acknowledges the GONG observing network and the MAGFiLO dataset. [Competition data](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
