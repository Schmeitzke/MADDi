import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageModel(nn.Module):
    """
    Convolutional neural network model for MRI image classification.
    """
    def __init__(self, in_channels=3, num_classes=3):
        """
        Initialize the model.
        
        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super(ImageModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, 100, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(100, 50, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Calculate the output size after the convolutional layers
        # For 72x72 input: after pooling twice (72/2/2 = 18), with 50 channels
        self.fc_input_size = 50 * 18 * 18
        
        # Fully connected layer
        self.classifier = nn.Linear(self.fc_input_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten the output for the fully connected layer
        features = x.view(x.size(0), -1)
        
        # Classification layer
        return self.classifier(features)
    
    def get_features(self, x):
        """
        Extract features from the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, fc_input_size).
        """
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten the output
        return x.view(x.size(0), -1) 