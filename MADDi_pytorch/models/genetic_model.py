import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneticModel(nn.Module):
    """
    Neural network model for genetic data (SNPs) classification.
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32, 32], dropout_rates=[0.5, 0.5, 0.3, 0.3], num_classes=3):
        """
        Initialize the model.
        
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of hidden layer sizes.
            dropout_rates (list): List of dropout rates for each hidden layer.
            num_classes (int): Number of output classes.
        """
        super(GeneticModel, self).__init__()
        
        layers = []
        in_features = input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """
        Extract features from the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, hidden_sizes[-1]).
        """
        return self.feature_extractor(x) 