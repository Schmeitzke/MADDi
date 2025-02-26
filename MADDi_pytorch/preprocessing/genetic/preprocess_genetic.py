import os
import argparse
import numpy as np
import pandas as pd
import gzip
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


def get_vcf_names(vcf_path):
    """
    Get column names from a VCF file.
    
    Args:
        vcf_path (str): Path to the VCF file.
        
    Returns:
        list: List of column names.
    """
    with gzip.open(vcf_path, "rt") as ifile:
        for line in ifile:
            if line.startswith("#CHROM"):
                vcf_names = [x for x in line.split('\t')]
                break
    return vcf_names


def read_vcf(path):
    """
    Read a VCF file into a pandas DataFrame.
    
    Args:
        path (str): Path to the VCF file.
        
    Returns:
        pd.DataFrame: VCF data.
    """
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def in_between(position, relevant):
    """
    Check if a position is between start and end positions in the relevant data.
    
    Args:
        position (int): Position to check.
        relevant (pd.DataFrame): Dataframe with start and end positions.
        
    Returns:
        bool: True if position is between start and end, False otherwise.
    """
    for i in range(len(relevant)):
        if (position >= relevant.iloc[i].start) and (position <= relevant.iloc[i].end):
            return True
    return False


def filter_vcfs(vcf_dir, gene_list_path, output_dir):
    """
    Filter VCF files based on gene positions.
    
    Args:
        vcf_dir (str): Directory containing VCF files.
        gene_list_path (str): Path to the gene list file.
        output_dir (str): Directory to save filtered VCF files.
        
    Returns:
        list: List of paths to the filtered VCF files.
    """
    print("Filtering VCF files...")
    genes = pd.read_csv(gene_list_path)
    files = os.listdir(vcf_dir)
    filtered_files = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for vcf_file in files:
        if not vcf_file.endswith(".vcf.gz"):
            continue
            
        file_name = os.path.join(vcf_dir, vcf_file)
        output_file_name = os.path.join(output_dir, vcf_file[:-7] + ".pkl")
        filtered_files.append(output_file_name)
        
        # Skip if the file already exists
        if os.path.exists(output_file_name):
            print(f"Skipping {vcf_file} as it's already processed")
            continue
        
        print(f"Processing {vcf_file}...")
        
        # Extract chromosome number from filename
        start = vcf_file.find("ADNI_ID.") + len("ADNI_ID.")
        end = vcf_file.find("output.vcf")
        chromosome = vcf_file[start:end]
        
        # Get relevant genes for this chromosome
        relevant = genes[genes["chrom"] == chromosome].reset_index()
        
        if len(relevant) == 0:
            print(f"No relevant genes found for chromosome {chromosome}")
            continue
        
        # Read VCF file
        try:
            names = get_vcf_names(file_name)
            vcf = pd.read_csv(file_name, compression='gzip', comment='#', 
                             delim_whitespace=True, header=None, names=names)
        except Exception as e:
            print(f"Error reading {vcf_file}: {e}")
            continue
        
        # Filter positions within gene boundaries
        positions = vcf["POS"]
        indexes = []
        
        for i in range(len(positions)):
            if i % 1000 == 0:
                print(f"  Checking position {i}/{len(positions)}")
                
            if in_between(positions[i], relevant):
                indexes.append(i)
        
        if len(indexes) > 0:
            df = vcf.iloc[indexes]
            df.to_pickle(output_file_name)
            print(f"Saved {len(indexes)} positions to {output_file_name}")
        else:
            print(f"No positions found within gene boundaries for {vcf_file}")
    
    return filtered_files


def concat_vcfs(filtered_files, diagnosis_path, output_path):
    """
    Concatenate filtered VCF files and merge with diagnosis data.
    
    Args:
        filtered_files (list): List of paths to filtered VCF files.
        diagnosis_path (str): Path to diagnosis data.
        output_path (str): Path to save the concatenated VCF file.
        
    Returns:
        pd.DataFrame: Concatenated VCF data.
    """
    print("Concatenating filtered VCF files...")
    diag = pd.read_csv(diagnosis_path)
    
    vcfs = []
    
    for vcf_file in filtered_files:
        print(f"Processing {vcf_file}...")
        vcf = pd.read_pickle(vcf_file)
        
        # Drop unnecessary columns
        vcf = vcf.drop(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'], axis=1)
        vcf = vcf.T
        vcf.reset_index(level=0, inplace=True)
        vcf["index"] = vcf["index"].str.replace("s", "S").str.replace("\n", "")
        
        # Merge with diagnosis data
        merged = diag.merge(vcf, on="index")
        merged = merged.rename(columns={"index": "subject"})
        
        # Convert genotypes to numeric values
        d = {'0/0': 0, '0/1': 1, '1/0': 1, '1/1': 2, "./.": 3}
        cols = list(set(merged.columns) - set(["subject", "Group"]))
        
        for idx, col in enumerate(cols):
            merged[col] = merged[col].str[:3].replace(d)
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(cols)} columns")
        
        # Add to list of dataframes
        vcfs.append(merged)
    
    # Concatenate all dataframes
    vcf = pd.concat(vcfs, ignore_index=True)
    vcf = vcf.drop_duplicates()
    vcf.to_pickle(output_path)
    
    return vcf


def feature_selection(vcf_data, overlap_test_path, output_dir):
    """
    Perform feature selection on genetic data using Random Forest.
    
    Args:
        vcf_data (pd.DataFrame): Concatenated VCF data.
        overlap_test_path (str): Path to overlap test set.
        output_dir (str): Directory to save processed data.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) Selected features and labels.
    """
    print("Performing feature selection...")
    
    # Make sure we have proper diagnosis values
    vcf_data = vcf_data[vcf_data["GROUP"] != -1]
    
    # Remove overlap test set if provided
    if overlap_test_path:
        print("Removing overlap test set...")
        ts = pd.read_csv(overlap_test_path)
        vcf1 = vcf_data[~vcf_data["subject"].isin(list(ts["subject"].values))]
    else:
        vcf1 = vcf_data
    
    # Prepare features and labels
    cols = list(set(vcf1.columns) - set(["subject", "Group", "GROUP", "label"]))
    X = vcf1[cols].values.astype(int)
    y = vcf1["GROUP"].astype(int).values - 1  # Convert to 0-based indexing
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    # Feature selection using Random Forest
    print("Training Random Forest for feature selection...")
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    sel.fit(X_train, y_train)
    
    # Get selected features
    selected_features = np.where(sel.get_support())[0]
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    
    # Apply feature selection
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Save selected data
    pd.DataFrame(X_train_selected).to_pickle(os.path.join(output_dir, "X_train_vcf.pkl"))
    pd.DataFrame(y_train).to_pickle(os.path.join(output_dir, "y_train_vcf.pkl"))
    pd.DataFrame(X_test_selected).to_pickle(os.path.join(output_dir, "X_test_vcf.pkl"))
    pd.DataFrame(y_test).to_pickle(os.path.join(output_dir, "y_test_vcf.pkl"))
    
    return X_train_selected, X_test_selected, y_train, y_test


def main(args):
    """
    Main function to run the genetic data preprocessing.
    
    Args:
        args: Command line arguments.
    """
    # Filter VCF files
    if args.filter_vcfs:
        filtered_files = filter_vcfs(args.vcf_dir, args.gene_list, args.filtered_dir)
    else:
        filtered_files = [os.path.join(args.filtered_dir, f) for f in os.listdir(args.filtered_dir) if f.endswith(".pkl")]
    
    # Concatenate filtered VCF files
    if args.concat_vcfs:
        vcf_data = concat_vcfs(filtered_files, args.diagnosis, args.output_vcf)
    else:
        vcf_data = pd.read_pickle(args.output_vcf)
    
    # Perform feature selection
    if args.feature_selection:
        X_train, X_test, y_train, y_test = feature_selection(vcf_data, args.overlap_test, args.output_dir)
        print(f"Final shapes: X_train {X_train.shape}, X_test {X_test.shape}")
    
    print("Genetic data preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess genetic data for Alzheimer's disease classification")
    parser.add_argument("--vcf_dir", type=str, help="Directory containing VCF files")
    parser.add_argument("--gene_list", type=str, help="Path to gene list CSV file")
    parser.add_argument("--filtered_dir", type=str, default="filtered_vcfs", help="Directory to save filtered VCF files")
    parser.add_argument("--diagnosis", type=str, help="Path to diagnosis file")
    parser.add_argument("--output_vcf", type=str, default="all_vcfs.pkl", help="Path to save concatenated VCF file")
    parser.add_argument("--overlap_test", type=str, help="Path to overlap test set file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save processed data")
    parser.add_argument("--filter_vcfs", action="store_true", help="Filter VCF files")
    parser.add_argument("--concat_vcfs", action="store_true", help="Concatenate filtered VCF files")
    parser.add_argument("--feature_selection", action="store_true", help="Perform feature selection")
    
    args = parser.parse_args()
    main(args) 