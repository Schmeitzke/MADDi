# MADDi PyTorch Implementation

This repository contains a PyTorch implementation of the Multimodal Alzheimer's Disease Diagnosis (MADDi) framework, which combines clinical data, genetic data (SNPs), and MRI images for Alzheimer's disease classification.

## Project Structure

```
MADDi_pytorch/
├── configs/               # Configuration files
├── data/                  # Data handling modules and classes
│   └── datasets.py        # PyTorch dataset classes for different modalities
├── models/                # Model definitions
│   ├── clinical_model.py  # Clinical data model
│   ├── genetic_model.py   # Genetic data model
│   ├── image_model.py     # MRI image model
│   └── multimodal_model.py # Multimodal model with attention mechanisms
├── preprocessing/         # Preprocessing scripts
│   ├── clinical/          # Clinical data preprocessing
│   ├── genetic/           # Genetic data preprocessing
│   ├── images/            # MRI image preprocessing
│   └── overlap/           # Preprocessing for multimodal data
├── training/              # Training scripts
│   ├── train_clinical.py  # Train clinical model
│   ├── train_genetic.py   # Train genetic model
│   ├── train_images.py    # Train image model
│   └── train_all_modalities.py # Train multimodal model
└── utils/                 # Utility functions
    └── utils.py           # General utility functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MADDi_pytorch.git
cd MADDi_pytorch
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The preprocessing scripts convert the raw data into a format suitable for training the models. Each modality has its own preprocessing pipeline.

### Clinical Data

```bash
python preprocessing/clinical/preprocess_clinical.py \
  --ground_truth path/to/ground_truth.csv \
  --demo path/to/PTDEMOG.csv \
  --neuro path/to/NEUROEXM.csv \
  --clinical path/to/ADSP_PHC_COGN.csv \
  --dxsum path/to/DXSUM_PDXCONV_ADNIALL.csv \
  --output_dir path/to/output
```

### Genetic Data

```bash
# Filter VCF files
python preprocessing/genetic/preprocess_genetic.py \
  --vcf_dir path/to/vcfs \
  --gene_list path/to/gene_list.csv \
  --filtered_dir filtered_vcfs \
  --filter_vcfs

# Concatenate filtered VCF files
python preprocessing/genetic/preprocess_genetic.py \
  --filtered_dir filtered_vcfs \
  --diagnosis path/to/diagnosis.csv \
  --output_vcf all_vcfs.pkl \
  --concat_vcfs

# Feature selection
python preprocessing/genetic/preprocess_genetic.py \
  --output_vcf all_vcfs.pkl \
  --output_dir path/to/output \
  --feature_selection
```

### Image Data

```bash
# Create dataset from raw images
python preprocessing/images/preprocess_images.py \
  --metadata path/to/metadata.csv \
  --data_dir path/to/images \
  --output_dataset mri_meta.pkl \
  --create_dataset

# Split dataset
python preprocessing/images/preprocess_images.py \
  --input_dataset mri_meta.pkl \
  --output_dir path/to/output \
  --split_dataset
```

### Overlap Dataset

```bash
python preprocessing/overlap/preprocess_overlap.py \
  --vcf path/to/all_vcfs.pkl \
  --clinical path/to/clinical.csv \
  --image path/to/mri_meta.pkl \
  --output_dir path/to/output
```

## Training

### Single Modality Models

#### Clinical Model

```bash
python training/train_clinical.py \
  --train_features path/to/X_train_c.pkl \
  --train_labels path/to/y_train_c.pkl \
  --test_features path/to/X_test_c.pkl \
  --test_labels path/to/y_test_c.pkl \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --epochs 100
```

#### Genetic Model

```bash
python training/train_genetic.py \
  --train_features path/to/X_train_vcf.pkl \
  --train_labels path/to/y_train_vcf.pkl \
  --test_features path/to/X_test_vcf.pkl \
  --test_labels path/to/y_test_vcf.pkl \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epochs 50
```

#### Image Model

```bash
python training/train_images.py \
  --train_images path/to/img_train.pkl \
  --train_labels path/to/img_y_train.pkl \
  --test_images path/to/img_test.pkl \
  --test_labels path/to/img_y_test.pkl \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epochs 50
```

### Multimodal Model

```bash
python training/train_all_modalities.py \
  --train_clinical path/to/X_train_clinical.pkl \
  --test_clinical path/to/X_test_clinical.pkl \
  --train_genetic path/to/X_train_snp.pkl \
  --test_genetic path/to/X_test_snp.pkl \
  --train_images path/to/X_train_img.pkl \
  --test_images path/to/X_test_img.pkl \
  --train_labels path/to/y_train.pkl \
  --test_labels path/to/y_test.pkl \
  --attention_mode MM_SA_BA \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epochs 50
```

## Available Attention Modes

The multimodal model supports the following attention modes:

- `MM_SA`: Self-attention for each modality
- `MM_BA`: Bi-directional cross-modal attention
- `MM_SA_BA`: Both self-attention and bi-directional cross-modal attention
- `None`: No attention, simple concatenation of features

## Models

### Clinical Model

The clinical model is a fully connected neural network with batch normalization and dropout layers.

### Genetic Model

The genetic model is a fully connected neural network with dropout layers.

### Image Model

The image model is a convolutional neural network (CNN) designed for MRI images.

### Multimodal Model

The multimodal model combines the features from the clinical, genetic, and image models using various attention mechanisms.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the original MADDi framework, which was developed for Alzheimer's disease classification using multiple modalities. 