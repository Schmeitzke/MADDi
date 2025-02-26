import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ClinicalDataset(Dataset):
    """
    Dataset class for clinical data.
    """
    def __init__(self, features, labels):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray or pd.DataFrame): Clinical features.
            labels (np.ndarray or pd.Series): Labels for each sample.
        """
        if isinstance(features, pd.DataFrame):
            self.features = features.values.astype(np.float32)
        else:
            self.features = features.astype(np.float32)
            
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.values.astype(np.int64).flatten()
        else:
            self.labels = labels.astype(np.int64).flatten()
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


class GeneticDataset(Dataset):
    """
    Dataset class for genetic data (SNPs).
    """
    def __init__(self, features, labels):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray or pd.DataFrame): Genetic features (SNPs).
            labels (np.ndarray or pd.Series): Labels for each sample.
        """
        if isinstance(features, pd.DataFrame):
            self.features = features.values.astype(np.float32)
        else:
            self.features = features.astype(np.float32)
            
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.values.astype(np.int64).flatten()
        else:
            self.labels = labels.astype(np.int64).flatten()
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


class ImageDataset(Dataset):
    """
    Dataset class for MRI image data.
    """
    def __init__(self, images, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images (np.ndarray): MRI image arrays.
            labels (np.ndarray or pd.Series): Labels for each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.values.astype(np.int64).flatten()
        else:
            self.labels = labels.astype(np.int64).flatten()
            
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        
        # Ensure correct format for PyTorch (C, H, W)
        if image.shape[-1] == 3:  # If the channels are in the last dimension
            image = np.transpose(image, (2, 0, 1))
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image), torch.tensor(self.labels[idx])


class MultimodalDataset(Dataset):
    """
    Dataset class for multimodal data (clinical, genetic, and imaging).
    """
    def __init__(self, clinical_features, genetic_features, images, labels):
        """
        Initialize the dataset.
        
        Args:
            clinical_features (np.ndarray or pd.DataFrame): Clinical features.
            genetic_features (np.ndarray or pd.DataFrame): Genetic features.
            images (np.ndarray): Image data.
            labels (np.ndarray or pd.Series): Labels for each sample.
        """
        # Convert pandas to numpy if needed
        if isinstance(clinical_features, pd.DataFrame):
            self.clinical_features = clinical_features.values.astype(np.float32)
        else:
            self.clinical_features = clinical_features.astype(np.float32)
            
        if isinstance(genetic_features, pd.DataFrame):
            self.genetic_features = genetic_features.values.astype(np.float32)
        else:
            self.genetic_features = genetic_features.astype(np.float32)
            
        self.images = images
        
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = labels.values.astype(np.int64).flatten()
        else:
            self.labels = labels.astype(np.int64).flatten()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        clinical = torch.tensor(self.clinical_features[idx])
        genetic = torch.tensor(self.genetic_features[idx])
        
        # Process image
        image = self.images[idx].astype(np.float32)
        if image.shape[-1] == 3:  # If the channels are in the last dimension
            image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)
        
        label = torch.tensor(self.labels[idx])
        
        return clinical, genetic, image, label 