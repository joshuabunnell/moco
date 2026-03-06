### 1. Data Strategy: Pretraining on Both Datasets

Pretrain on both datasets. CT physics are consistent across age groups; scanner noise and tissue types are similar. Pediatric data helps the model learn CT-specific features. Main domain shift: adults have more visceral fat and ACRIN 6664 uses insufflation (air in colon), while pediatric scans do not. Fine-tuning must use only ACRIN 6664.

---

### 2. Workflow Overview

**Phase 1: Data Preparation**
- Write a Python script (not a notebook) using the finalized `prep_transforms` pipeline.
- Run on HPC to process both datasets.
- Save each scan as a 1mm isotropic, windowed 3D PyTorch tensor (`.pt`) to SSD.

**Phase 2: Unsupervised Pretraining (MoCo v2)**
- Dataloader samples random 2.5D crops from both datasets, applies augmentations, and forms positive pairs.
- Model learns CT tissue features without labels.
- Output: ResNet-50 encoder with CT-specific weights (discard MoCo projection head).

**Phase 3: Supervised Fine-Tuning**
- Attach new classification head to pretrained ResNet-50.
- Use only ACRIN 6664 data and clinical labels.
- Train with Cross-Entropy Loss. Optionally freeze backbone or fine-tune at low learning rate.

---

### 3. SSL Pretraining Evaluation

- **InfoNCE Loss:** Should decrease and stabilize. Rapid drop then plateau is expected. Oscillation or instant zero indicates issues with augmentations or learning rate.
- **KNN Accuracy:** Freeze model, extract features, run KNN on labeled subset. Accuracy should rise as features improve.
- **Linear Probing:** After pretraining, freeze encoder, train a linear layer on ACRIN labels. High accuracy means good features; low accuracy means pretraining or augmentations need adjustment.

---

**Next Step:**
Wrap preprocessing code into a `.py` script, set up HPC, and convert DICOMs to `.pt` tensors. Proceed to MoCo training after this step.