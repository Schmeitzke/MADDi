import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import pickle5 as pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import gc

import sys
sys.path.append('..')
from models.multimodal_model import MultimodalModel
from data.datasets import MultimodalDataset
from utils.utils import set_seed, calc_confusion_matrix, make_img


def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    """
    Train the multimodal model.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epochs (int): Number of epochs to train for.
        
    Returns:
        dict: Training history.
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for clinical, genetic, images, labels in train_loader:
            clinical, genetic, images, labels = (
                clinical.to(device), 
                genetic.to(device), 
                images.to(device), 
                labels.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model(clinical, genetic, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * clinical.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for clinical, genetic, images, labels in valid_loader:
                clinical, genetic, images, labels = (
                    clinical.to(device), 
                    genetic.to(device), 
                    images.to(device), 
                    labels.to(device)
                )
                
                outputs = model(clinical, genetic, images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * clinical.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(valid_loader.dataset)
        val_accuracy = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return history


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): Model to evaluate.
        test_loader (DataLoader): Test data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        
    Returns:
        tuple: (test_loss, test_accuracy, predictions, true_labels)
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for clinical, genetic, images, labels in test_loader:
            clinical, genetic, images, labels = (
                clinical.to(device), 
                genetic.to(device), 
                images.to(device), 
                labels.to(device)
            )
            
            outputs = model(clinical, genetic, images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * clinical.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    
    # Concatenate batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return test_loss, test_accuracy, all_predictions, all_labels


def plot_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main(args):
    """
    Main function to run the training and evaluation.
    
    Args:
        args: Command line arguments.
    """
    # Set up seeds for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load clinical data
    X_train_clinical = pd.read_pickle(args.train_clinical)
    X_test_clinical = pd.read_pickle(args.test_clinical)
    
    # Load genetic data
    X_train_genetic = pd.read_pickle(args.train_genetic)
    X_test_genetic = pd.read_pickle(args.test_genetic)
    
    # Load image data
    X_train_image = pd.read_pickle(args.train_images)
    X_test_image = pd.read_pickle(args.test_images)
    
    # Load labels
    y_train = pd.read_pickle(args.train_labels)
    y_test = pd.read_pickle(args.test_labels)
    
    # Convert image data to numpy arrays if necessary
    if isinstance(X_train_image, pd.DataFrame):
        X_train_image_np = []
        for i in range(len(X_train_image)):
            X_train_image_np.append(X_train_image.values[i][0])
        X_train_image_np = np.array(X_train_image_np)
    else:
        X_train_image_np = X_train_image
    
    if isinstance(X_test_image, pd.DataFrame):
        X_test_image_np = []
        for i in range(len(X_test_image)):
            X_test_image_np.append(X_test_image.values[i][0])
        X_test_image_np = np.array(X_test_image_np)
    else:
        X_test_image_np = X_test_image
    
    # Create datasets
    train_dataset = MultimodalDataset(X_train_clinical, X_train_genetic, X_train_image_np, y_train)
    test_dataset = MultimodalDataset(X_test_clinical, X_test_genetic, X_test_image_np, y_test)
    
    # Split training set for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = MultimodalModel(
        clinical_input_size=X_train_clinical.shape[1],
        genetic_input_size=X_train_genetic.shape[1],
        mode=args.attention_mode,
        num_classes=3
    )
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    history = train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
    
    # Plot training history
    plot_history(
        history, 
        save_path=f'multimodal_{args.attention_mode}_lr{args.learning_rate}_bs{args.batch_size}_epochs{args.epochs}.png'
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, test_predictions, test_labels = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # Calculate and print classification report
    cr, precision, recall, thresholds = calc_confusion_matrix(
        test_predictions, test_labels, args.attention_mode, args.learning_rate, args.batch_size, args.epochs
    )
    
    # Save model
    torch.save(
        model.state_dict(), 
        f'multimodal_{args.attention_mode}_lr{args.learning_rate}_bs{args.batch_size}_epochs{args.epochs}.pth'
    )
    
    # Clean up to release GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return results for multiple runs
    return {
        'accuracy': test_accuracy,
        'precision': cr['macro avg']['precision'],
        'recall': cr['macro avg']['recall'],
        'f1-score': cr['macro avg']['f1-score']
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multimodal model')
    parser.add_argument('--train_clinical', type=str, default='X_train_clinical.pkl', help='Path to training clinical features')
    parser.add_argument('--test_clinical', type=str, default='X_test_clinical.pkl', help='Path to test clinical features')
    parser.add_argument('--train_genetic', type=str, default='X_train_snp.pkl', help='Path to training genetic features')
    parser.add_argument('--test_genetic', type=str, default='X_test_snp.pkl', help='Path to test genetic features')
    parser.add_argument('--train_images', type=str, default='X_train_img.pkl', help='Path to training images')
    parser.add_argument('--test_images', type=str, default='X_test_img.pkl', help='Path to test images')
    parser.add_argument('--train_labels', type=str, default='y_train.pkl', help='Path to training labels')
    parser.add_argument('--test_labels', type=str, default='y_test.pkl', help='Path to test labels')
    parser.add_argument('--attention_mode', type=str, default='MM_SA_BA', choices=['MM_SA', 'MM_BA', 'MM_SA_BA', 'None'], 
                       help='Attention mode (MM_SA=self-attention, MM_BA=bi-directional attention, MM_SA_BA=both, None=no attention)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs with different seeds')
    
    args = parser.parse_args()
    
    if args.num_runs > 1:
        # Multiple runs with different seeds
        accuracy = []
        precision = []
        recall = []
        f1_score = []
        seeds = random.sample(range(1, 200), args.num_runs)
        
        for seed in seeds:
            args.seed = seed
            result = main(args)
            accuracy.append(result['accuracy'])
            precision.append(result['precision'])
            recall.append(result['recall'])
            f1_score.append(result['f1-score'])
        
        # Print average results
        print("\nAverage Results:")
        print(f"Attention Mode: {args.attention_mode}")
        print(f"Accuracy: {np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}")
        print(f"Precision: {np.mean(precision):.4f} ± {np.std(precision):.4f}")
        print(f"Recall: {np.mean(recall):.4f} ± {np.std(recall):.4f}")
        print(f"F1-score: {np.mean(f1_score):.4f} ± {np.std(f1_score):.4f}")
        
        # Print all results
        print("\nAll Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1_score}")
    else:
        # Single run
        main(args) 