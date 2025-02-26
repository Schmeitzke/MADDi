import argparse
import numpy as np
import skimage.transform as skTrans
import nibabel as nib
import pandas as pd
import os
import time
import pickle
from sklearn.model_selection import train_test_split


def normalize_img(img_array):
    """
    Normalize image array by dividing by the 99.5th percentile.
    
    Args:
        img_array (np.ndarray): Image array to normalize.
        
    Returns:
        np.ndarray: Normalized image array.
    """
    maxes = np.quantile(img_array, 0.995, axis=(0, 1, 2))
    return img_array / maxes


def process_image(path, img_id, meta):
    """
    Process a single MRI image file.
    
    Args:
        path (str): Path to the image file.
        img_id (str): Image ID.
        meta (pd.DataFrame): Metadata for images.
        
    Returns:
        tuple: (processed_image, label, subject) Processed image, label, and subject ID.
    """
    idx = meta[meta["Image Data ID"] == img_id].index[0]
    
    # Load and preprocess the image
    try:
        im = nib.load(path).get_fdata()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None
    
    # Get the center slices
    n_i, n_j, n_k = im.shape
    center_i = (n_i - 1) // 2  
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    
    # Resize slices to 72x72
    im1 = skTrans.resize(im[center_i, :, :], (72, 72), order=1, preserve_range=True)
    im2 = skTrans.resize(im[:, center_j, :], (72, 72), order=1, preserve_range=True)
    im3 = skTrans.resize(im[:, :, center_k], (72, 72), order=1, preserve_range=True)
    
    # Stack slices as channels
    im = np.array([im1, im2, im3]).T
    
    # Get label and subject information
    label = meta.at[idx, "Group"]
    subject = meta.at[idx, "Subject"]
    
    # Normalize image
    norm_im = normalize_img(im)
    
    return norm_im, label, subject


def create_dataset(metadata_path, data_dir, output_path):
    """
    Create a dataset of processed MRI images.
    
    Args:
        metadata_path (str): Path to the metadata CSV file.
        data_dir (str): Directory containing MRI image files.
        output_path (str): Path to save the dataset.
        
    Returns:
        pd.DataFrame: Dataframe with image arrays, labels, and subject IDs.
    """
    print("Loading metadata...")
    meta = pd.read_csv(metadata_path)
    
    # Keep only necessary columns
    meta = meta[["Image Data ID", "Group", "Subject"]]
    
    # Convert group labels to numeric
    meta["Group"] = pd.factorize(meta["Group"])[0]
    
    # Initialize dataframe for processed images
    meta_all = pd.DataFrame(columns=["img_array", "label", "subject"])
    
    # Process each image
    files = os.listdir(data_dir)
    file_count = len(files)
    print(f"Processing {file_count} image files...")
    
    for i, file in enumerate(files):
        if file == '.DS_Store':
            continue
            
        path = os.path.join(data_dir, file)
        
        # Extract image ID from filename
        start = '_'
        end = '.nii'
        img_id = file.split(start)[-1].split(end)[0]
        
        # Process image
        img_array, label, subject = process_image(path, img_id, meta)
        
        if img_array is not None:
            # Add to dataset
            meta_all = meta_all.append({"img_array": img_array, "label": label, "subject": subject}, ignore_index=True)
            
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == file_count:
            print(f"Processed {i + 1}/{file_count} images")
    
    # Save dataset
    print(f"Saving dataset with {len(meta_all)} processed images to {output_path}")
    meta_all.to_pickle(output_path)
    
    return meta_all


def split_dataset(dataset_path, output_dir, test_size=0.1, random_state=42, overlap_test_path=None):
    """
    Split the dataset into training and testing sets.
    
    Args:
        dataset_path (str): Path to the dataset pickle file.
        output_dir (str): Directory to save the split datasets.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        overlap_test_path (str): Path to overlap test set file.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) Training and testing data.
    """
    print("Loading dataset...")
    m2 = pd.read_pickle(dataset_path)
    
    # Clean subject IDs
    m2["subject"] = m2["subject"].str.replace("s", "S").str.replace("\n", "")
    
    # Remove overlap test set if provided
    if overlap_test_path:
        print("Removing overlap test set...")
        ts = pd.read_csv(overlap_test_path)
        m2 = m2[~m2["subject"].isin(list(ts["subject"].values))]
    
    # Get unique subjects
    subjects = list(set(m2["subject"].values))
    print(f"Dataset contains {len(subjects)} unique subjects and {len(m2)} images")
    
    # Pick subjects for testing
    num_test_subjects = int(len(subjects) * test_size)
    np.random.seed(random_state)
    test_subjects = np.random.choice(subjects, num_test_subjects, replace=False)
    
    # Create test set with one image per subject
    test = pd.DataFrame(columns=["img_array", "subject", "label"])
    for subject in test_subjects:
        # Get one random image for each test subject
        s = m2[m2["subject"] == subject].sample(1, random_state=random_state)
        test = test.append(s)
    
    # All other images go to training set
    train = m2[~m2.index.isin(test.index)]
    
    print(f"Split dataset into {len(train)} training and {len(test)} testing images")
    
    # Extract arrays and labels
    X_train = np.array([x for x in train["img_array"].values])
    y_train = train["label"].values
    
    X_test = np.array([x for x in test["img_array"].values])
    y_test = test["label"].values
    
    # Save split datasets
    print("Saving split datasets...")
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame({"img_array": train["img_array"]}).to_pickle(os.path.join(output_dir, "img_train.pkl"))
    pd.DataFrame({"label": train["label"]}).to_pickle(os.path.join(output_dir, "img_y_train.pkl"))
    pd.DataFrame({"img_array": test["img_array"]}).to_pickle(os.path.join(output_dir, "img_test.pkl"))
    pd.DataFrame({"label": test["label"]}).to_pickle(os.path.join(output_dir, "img_y_test.pkl"))
    
    return X_train, X_test, y_train, y_test


def main(args):
    """
    Main function to run the image data preprocessing.
    
    Args:
        args: Command line arguments.
    """
    # Create dataset if requested
    if args.create_dataset:
        create_dataset(args.metadata, args.data_dir, args.output_dataset)
    
    # Split dataset if requested
    if args.split_dataset:
        split_dataset(args.input_dataset, args.output_dir, args.test_size, args.random_state, args.overlap_test)
    
    print("Image data preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRI images for Alzheimer's disease classification")
    parser.add_argument("--metadata", type=str, help="Path to metadata CSV file")
    parser.add_argument("--data_dir", type=str, help="Directory containing MRI image files")
    parser.add_argument("--output_dataset", type=str, default="mri_meta.pkl", help="Path to save the processed dataset")
    parser.add_argument("--input_dataset", type=str, default="mri_meta.pkl", help="Path to the input dataset for splitting")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save split datasets")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--overlap_test", type=str, help="Path to overlap test set file")
    parser.add_argument("--create_dataset", action="store_true", help="Create dataset from raw images")
    parser.add_argument("--split_dataset", action="store_true", help="Split dataset into training and testing sets")
    
    args = parser.parse_args()
    main(args) 