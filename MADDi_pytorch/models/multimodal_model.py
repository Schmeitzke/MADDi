import torch
import torch.nn as nn
import torch.nn.functional as F
from .clinical_model import ClinicalModel
from .genetic_model import GeneticModel
from .image_model import ImageModel


class MultiHeadAttention(nn.Module):
    """
    PyTorch implementation of Multi-Head Attention.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the multi-head attention module.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor, optional): Attention mask.
            
        Returns:
            torch.Tensor: Output tensor after attention.
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_linear(output)


class MultimodalModel(nn.Module):
    """
    Multimodal model that combines clinical, genetic, and imaging data with attention mechanisms.
    """
    def __init__(self, clinical_input_size, genetic_input_size, mode='MM_SA_BA', num_classes=3):
        """
        Initialize the multimodal model.
        
        Args:
            clinical_input_size (int): Number of clinical features.
            genetic_input_size (int): Number of genetic features.
            mode (str): Attention mode - 'MM_SA' (self-attention), 'MM_BA' (bi-directional attention),
                        'MM_SA_BA' (both), or 'None' (no attention).
            num_classes (int): Number of output classes.
        """
        super(MultimodalModel, self).__init__()
        
        # Feature extractors
        self.clinical_model = ClinicalModel(clinical_input_size)
        self.genetic_model = GeneticModel(genetic_input_size)
        self.image_model = ImageModel()
        
        # Feature dimensions
        self.clinical_feature_dim = 50
        self.genetic_feature_dim = 32
        self.image_feature_dim = self.image_model.fc_input_size
        
        # Attention mechanisms
        self.mode = mode
        
        if mode in ['MM_SA', 'MM_SA_BA']:
            # Self-attention mechanisms
            self.clinical_self_attn = MultiHeadAttention(self.clinical_feature_dim, num_heads=4)
            self.genetic_self_attn = MultiHeadAttention(self.genetic_feature_dim, num_heads=4)
            self.image_self_attn = MultiHeadAttention(self.image_feature_dim, num_heads=4)
        
        if mode in ['MM_BA', 'MM_SA_BA']:
            # Cross-modal attention mechanisms
            self.image_clinical_attn = MultiHeadAttention(self.clinical_feature_dim, num_heads=4)
            self.genetic_image_attn = MultiHeadAttention(self.image_feature_dim, num_heads=4)
            self.clinical_genetic_attn = MultiHeadAttention(self.genetic_feature_dim, num_heads=4)
        
        # Determine the size of the concatenated features
        if mode == 'MM_SA':
            concat_dim = (self.clinical_feature_dim + self.genetic_feature_dim + 
                          self.image_feature_dim) * 2  # Original + self-attention
        elif mode == 'MM_BA':
            concat_dim = (self.clinical_feature_dim + self.genetic_feature_dim + 
                          self.image_feature_dim) * 2  # Original + cross-attention
        elif mode == 'MM_SA_BA':
            concat_dim = (self.clinical_feature_dim + self.genetic_feature_dim + 
                          self.image_feature_dim) * 2  # Original + cross of self-attention
        else:  # 'None'
            concat_dim = self.clinical_feature_dim + self.genetic_feature_dim + self.image_feature_dim
            
        # Classifier
        self.classifier = nn.Linear(concat_dim, num_classes)
        
    def forward(self, clinical, genetic, image):
        """
        Forward pass of the multimodal model.
        
        Args:
            clinical (torch.Tensor): Clinical features.
            genetic (torch.Tensor): Genetic features.
            image (torch.Tensor): Image data.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Extract features from each modality
        clinical_features = self.clinical_model.get_features(clinical)
        genetic_features = self.genetic_model.get_features(genetic)
        image_features = self.image_model.get_features(image)
        
        batch_size = clinical.size(0)
        
        # Process based on attention mode
        if self.mode == 'MM_SA':
            # Self-attention
            clinical_attn = self._apply_self_attention(clinical_features, self.clinical_self_attn)
            genetic_attn = self._apply_self_attention(genetic_features, self.genetic_self_attn)
            image_attn = self._apply_self_attention(image_features, self.image_self_attn)
            
            # Concatenate original features and self-attention features
            concat_features = torch.cat([
                clinical_features, genetic_features, image_features,
                clinical_attn, genetic_attn, image_attn
            ], dim=1)
            
        elif self.mode == 'MM_BA':
            # Cross-modal bi-directional attention
            image_clinical_attn = self._apply_cross_attention(
                image_features, clinical_features, self.image_clinical_attn)
            genetic_image_attn = self._apply_cross_attention(
                genetic_features, image_features, self.genetic_image_attn)
            clinical_genetic_attn = self._apply_cross_attention(
                clinical_features, genetic_features, self.clinical_genetic_attn)
            
            # Concatenate original features and cross-attention features
            concat_features = torch.cat([
                clinical_features, genetic_features, image_features,
                image_clinical_attn, genetic_image_attn, clinical_genetic_attn
            ], dim=1)
            
        elif self.mode == 'MM_SA_BA':
            # Self-attention first
            clinical_self_attn = self._apply_self_attention(clinical_features, self.clinical_self_attn)
            genetic_self_attn = self._apply_self_attention(genetic_features, self.genetic_self_attn)
            image_self_attn = self._apply_self_attention(image_features, self.image_self_attn)
            
            # Cross-modal bi-directional attention on self-attention features
            image_clinical_attn = self._apply_cross_attention(
                image_self_attn, clinical_self_attn, self.image_clinical_attn)
            genetic_image_attn = self._apply_cross_attention(
                genetic_self_attn, image_self_attn, self.genetic_image_attn)
            clinical_genetic_attn = self._apply_cross_attention(
                clinical_self_attn, genetic_self_attn, self.clinical_genetic_attn)
            
            # Concatenate original features and attention features
            concat_features = torch.cat([
                clinical_features, genetic_features, image_features,
                image_clinical_attn, genetic_image_attn, clinical_genetic_attn
            ], dim=1)
            
        else:  # 'None'
            # Simple concatenation of features
            concat_features = torch.cat([
                clinical_features, genetic_features, image_features
            ], dim=1)
        
        # Classification
        return self.classifier(concat_features)
    
    def _apply_self_attention(self, features, attention_module):
        """
        Apply self-attention to features.
        
        Args:
            features (torch.Tensor): Input features.
            attention_module (nn.Module): Attention module.
            
        Returns:
            torch.Tensor: Features after self-attention.
        """
        batch_size = features.size(0)
        # Add sequence dimension for attention
        features_seq = features.unsqueeze(1)
        # Apply self-attention
        attended = attention_module(features_seq, features_seq, features_seq)
        # Remove sequence dimension
        return attended.squeeze(1)
    
    def _apply_cross_attention(self, query_features, key_value_features, attention_module):
        """
        Apply cross-attention between two feature sets.
        
        Args:
            query_features (torch.Tensor): Query features.
            key_value_features (torch.Tensor): Key and value features.
            attention_module (nn.Module): Attention module.
            
        Returns:
            torch.Tensor: Features after cross-attention.
        """
        batch_size = query_features.size(0)
        # Add sequence dimension for attention
        query_seq = query_features.unsqueeze(1)
        key_value_seq = key_value_features.unsqueeze(1)
        # Apply cross-attention
        attended = attention_module(query_seq, key_value_seq, key_value_seq)
        # Remove sequence dimension
        return attended.squeeze(1) 