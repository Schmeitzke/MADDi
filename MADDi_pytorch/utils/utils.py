import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_img(img_path):
    """
    Load image data from pickle file.
    
    Args:
        img_path (str): Path to the pickle file containing images.
        
    Returns:
        np.ndarray: Array of image data.
    """
    img = pd.read_pickle(img_path)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    
    return np.array(img_l)


def plot_classification_report(y_true, y_pred, mode, learning_rate, batch_size, epochs, figsize=(7, 7), ax=None):
    """
    Plot classification report as a heatmap.
    
    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        mode (str): Training mode description.
        learning_rate (float): Learning rate used.
        batch_size (int): Batch size used.
        epochs (int): Number of epochs trained.
        figsize (tuple): Figure size (width, height).
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's"]
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_true, y_pred)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap="Blues")
    
    plt.savefig(f'report_{mode}_{learning_rate}_{batch_size}_{epochs}.png')


def calc_confusion_matrix(outputs, labels, mode, learning_rate, batch_size, epochs):
    """
    Calculate confusion matrix and precision-recall metrics.
    
    Args:
        outputs (torch.Tensor): Model outputs (probabilities).
        labels (torch.Tensor): True labels.
        mode (str): Training mode description.
        learning_rate (float): Learning rate used.
        batch_size (int): Batch size used.
        epochs (int): Number of epochs trained.
        
    Returns:
        tuple: (classification_report, precision_dict, recall_dict, thresholds_dict)
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Convert to one-hot encoding if needed
    if len(labels.shape) == 1:
        n_classes = outputs.shape[1]
        labels_one_hot = np.zeros((labels.size, n_classes))
        labels_one_hot[np.arange(labels.size), labels] = 1
    else:
        labels_one_hot = labels
        
    true_label = np.argmax(labels_one_hot, axis=1)
    predicted_label = np.argmax(outputs, axis=1)
    
    n_classes = 3
    precision = {}
    recall = {}
    thres = {}
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(
            labels_one_hot[:, i], outputs[:, i])

    print("Classification Report:")
    print(classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    
    return cr, precision, recall, thres


def save_model(model, path):
    """
    Save PyTorch model to disk.
    
    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load model weights from disk.
    
    Args:
        model (torch.nn.Module): Model architecture to load weights into.
        path (str): Path to the saved weights.
        
    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    model.load_state_dict(torch.load(path))
    return model 