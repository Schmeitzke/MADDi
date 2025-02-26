import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_data(vcf_path, clinical_path, image_path):
    """
    Load data from different modalities.
    
    Args:
        vcf_path (str): Path to genetic data pickle file.
        clinical_path (str): Path to clinical data CSV file.
        image_path (str): Path to image data pickle file.
        
    Returns:
        tuple: (vcf, clinical, image) Dataframes for each modality.
    """
    print("Loading data from different modalities...")
    
    # Load genetic data
    vcf = pd.read_pickle(vcf_path)
    
    # Load clinical data
    c = pd.read_csv(clinical_path)
    if "Unnamed: 0" in c.columns:
        c = c.drop("Unnamed: 0", axis=1)
    c = c.rename(columns={"PTID": "subject"})
    
    # Load image data
    img = pd.read_pickle(image_path)
    
    print(f"Loaded data: {len(vcf)} genetic samples, {len(c)} clinical samples, {len(img)} images")
    
    return vcf, c, img


def merge_modalities(vcf, clinical, image):
    """
    Merge data from different modalities.
    
    Args:
        vcf (pd.DataFrame): Genetic data.
        clinical (pd.DataFrame): Clinical data.
        image (pd.DataFrame): Image data.
        
    Returns:
        pd.DataFrame: Merged data.
    """
    print("Merging modalities...")
    
    # Rename GROUP column in clinical data if needed
    if "Group" in clinical.columns and "GROUP" not in clinical.columns:
        clinical = clinical.rename(columns={"Group": "GROUP"})
    
    # Merge all modalities
    a = vcf.merge(clinical, on=["subject", "GROUP"]).merge(image, on="subject")
    
    print(f"After merging, {len(a)} samples remain with data from all modalities")
    print(f"Subject distribution: {a['subject'].value_counts().describe()}")
    print(f"Diagnosis distribution: {a['GROUP'].value_counts()}")
    
    return a


def split_and_save_data(merged_data, output_dir, test_size=0.1, random_state=42):
    """
    Split data into training and testing sets and save.
    
    Args:
        merged_data (pd.DataFrame): Merged data from all modalities.
        output_dir (str): Directory to save processed data.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for train-test split.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) Split data.
    """
    print("Preparing to split data...")
    
    # Prepare features and labels
    cols = list(set(merged_data.columns) - set(["PTID", "label", "GROUP", "RID", "ID", "Group", "Phase", "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2", "update_stamp", "DX", "Unnamed: 0"]))
    X = merged_data[cols]
    y = merged_data["GROUP"]
    
    # Split into train and test sets
    print(f"Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test subjects for future reference
    X_test[["subject"]].to_csv(os.path.join(output_dir, "overlap_test_set.csv"))
    
    # Extract modality-specific features
    print("Extracting and saving modality-specific data...")
    
    # Identify columns for each modality
    snp_cols = set(X_train.columns).intersection(set(vcf.columns))
    img_cols = set(X_train.columns).intersection(set(img.columns))
    clin_cols = set(X_train.columns).intersection(set(clinical.columns))
    
    print(f"Feature counts: {len(snp_cols)} genetic, {len(img_cols)} image, {len(clin_cols)} clinical")
    
    # Extract modality-specific features
    X_train_snp = X_train[snp_cols]
    X_test_snp = X_test[snp_cols]
    
    X_train_img = X_train[img_cols]
    X_test_img = X_test[img_cols]
    
    X_train_clin = X_train[clin_cols]
    X_test_clin = X_test[clin_cols]
    
    # Save to files
    pd.DataFrame(X_train_snp).to_pickle(os.path.join(output_dir, "X_train_snp.pkl"))
    pd.DataFrame(X_test_snp).to_pickle(os.path.join(output_dir, "X_test_snp.pkl"))
    pd.DataFrame(y_train).to_pickle(os.path.join(output_dir, "y_train.pkl"))
    pd.DataFrame(y_test).to_pickle(os.path.join(output_dir, "y_test.pkl"))
    
    pd.DataFrame(X_train_clin).to_pickle(os.path.join(output_dir, "X_train_clinical.pkl"))
    pd.DataFrame(X_test_clin).to_pickle(os.path.join(output_dir, "X_test_clinical.pkl"))
    
    pd.DataFrame(X_train_img).to_pickle(os.path.join(output_dir, "X_train_img.pkl"))
    pd.DataFrame(X_test_img).to_pickle(os.path.join(output_dir, "X_test_img.pkl"))
    
    print(f"All data successfully processed and saved to {output_dir}")
    
    return X_train, X_test, y_train, y_test


def main(args):
    """
    Main function to run the overlap dataset preprocessing.
    
    Args:
        args: Command line arguments.
    """
    # Load data from each modality
    vcf, clinical, img = load_data(args.vcf, args.clinical, args.image)
    
    # Merge modalities
    merged_data = merge_modalities(vcf, clinical, img)
    
    # Split and save data
    X_train, X_test, y_train, y_test = split_and_save_data(
        merged_data, args.output_dir, args.test_size, args.random_state
    )
    
    print("Overlap dataset preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess overlap dataset for multimodal Alzheimer's disease classification")
    parser.add_argument("--vcf", type=str, required=True, help="Path to genetic data pickle file")
    parser.add_argument("--clinical", type=str, required=True, help="Path to clinical data CSV file")
    parser.add_argument("--image", type=str, required=True, help="Path to image data pickle file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for train-test split")
    
    args = parser.parse_args()
    main(args) 