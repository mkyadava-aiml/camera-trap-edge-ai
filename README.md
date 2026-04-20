# camera-trap-edge-ai
A Comparative Evaluation of Transfer-Learned Edge AI Detection Models for Camera Trap Wildlife Monitoring Under Diverse Environmental Conditions Using Balanced and Augmented Public Datasets
(Data-centric evaluation of edge AI models for camera trap wildlife monitoring)
# Camera Trap Edge AI: Data-Centric Evaluation Framework

## Overview

This repository presents a data-centric evaluation framework for transfer-learned edge AI object detection models applied to wildlife monitoring using camera trap imagery.

Unlike conventional approaches that focus primarily on model architecture, this work emphasizes the role of **data pipeline design** in determining model performance, robustness, and generalization. 
The study systematically evaluates how factors such as illumination variation, location bias, sequence dependence, and class imbalance influence detection outcomes under realistic ecological conditions.

---

## Key Contributions

- A **reproducible camera trap data pipeline formalism**
- Event-level dataset construction to eliminate sequence redundancy
- Location-aware train–test splitting to ensure true generalization
- Controlled dataset generation aligned with research questions (RQ1–RQ6)
- Evaluation of edge AI detection models under realistic deployment conditions

---

## Dataset

The dataset is derived from publicly available camera trap repositories (e.g., WCS / LILA).

### Key Characteristics:
- Wildlife images with bounding box annotations
- Metadata: location, timestamp, sequence ID
- Event-level grouping (burst-based)
- Derived attributes:
  - illumination (day/night)
  - class grouping (common vs rare)
  - location-based splits

## **Note:**  
Due to size constraints, raw datasets are not included in this repository. 
Only sample metadata and split definitions are provided.

---

## Data Pipeline

The dataset preparation pipeline consists of four stages:

### 1. Manifest Construction
- Class-balanced sampling
- Sequence-aware filtering
- Empty frame control  # Only empty images (no humans)

### 2. Data Download
- Parallel image retrieval
- Multi-source fallback support

### 3. Dataset Splitting
- Illumination-based splits (RQ1)
- Location-based splits (RQ2)
- Sequence-controlled datasets (RQ3)
- Class grouping (RQ4)
- Background diversity control (RQ5)
## Gray images added separately whenever image timestamp showed day but image was grayscale

### 4. Dataset Transformation
- Image resizing and normalization
- Grid-based target encoding for detection models

---

## Research Questions (RQs)
RQ1: Illumination robustness (day vs night)
RQ2: Location generalization
RQ3: Sequence-level data leakage
RQ4: Class imbalance effect
RQ5: Background repetition bias
RQ6: Overall pipeline robustness

---

## Repository Structure

- see image_directory_struct.png


---

## Installation

```bash
pip install -r requirements.txt    #not provided here. Its a placeholder for future works
Usage
Step 1: Build dataset manifest
python3 src/preprocessing/build_manifest.py
Step 2: Download images
python3 src/preprocessing/download_data.py
Step 3: Create dataset splits
python3 src/preprocessing/create_splits.py
Step 4: Build ML dataset
python3 src/preprocessing/build_dataset.py
Evaluation Metrics

#Model performance is evaluated using:

-Precision
-Recall
-F1-score
-mAP@0.50
-mAP@0.50–0.95
-Intersection over Union (IoU)
-Reproducibility

This repository ensures reproducibility through:

Explicit dataset split definitions (CSV files)
Sequence-level de-duplication
Location-aware partitioning
Fully documented preprocessing pipeline
Notes
Raw images and large datasets are excluded due to storage constraints
Users are expected to download source datasets independently
Sample metadata and split definitions are included for reference
Citation

If you use this work, please cite:

(Yadava M.K., 2026)
Camera Trap Edge AI: A Comparative Evaluation of Transfer-Learned Edge AI Detection Models for Camera Trap Wildlife Monitoring Under Diverse Environmental Conditions Using Balanced and Augmented Public Datasets

License

This project is intended for academic and research use.
