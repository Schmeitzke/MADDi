import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(ground_truth_path, demo_path, neuro_path, clinical_path, dxsum_path):
    """
    Load and merge clinical data from different sources.
    
    Args:
        ground_truth_path (str): Path to ground truth diagnosis file.
        demo_path (str): Path to demographic data.
        neuro_path (str): Path to neurological examination data.
        clinical_path (str): Path to clinical data.
        dxsum_path (str): Path to diagnosis summary data.
        
    Returns:
        pd.DataFrame: Merged clinical data.
    """
    print("Loading data...")
    # Load ground truth diagnosis
    diag = pd.read_csv(ground_truth_path).drop("Unnamed: 0", axis=1) if "Unnamed: 0" in pd.read_csv(ground_truth_path).columns else pd.read_csv(ground_truth_path)
    
    # Load clinical data
    demo = pd.read_csv(demo_path)
    neuro = pd.read_csv(neuro_path)
    clinical = pd.read_csv(clinical_path).rename(columns={"PHASE": "Phase"})
    comb = pd.read_csv(dxsum_path)[["RID", "PTID", "Phase"]]
    
    # Merge all data
    print("Merging data...")
    m = comb.merge(demo, on=["RID", "Phase"]).merge(neuro, on=["RID", "Phase"]).merge(clinical, on=["RID", "Phase"]).drop_duplicates()
    
    # Remove duplicate columns
    m.columns = [c[:-2] if str(c).endswith(('_x', '_y')) else c for c in m.columns]
    m = m.loc[:, ~m.columns.duplicated()]
    
    # Merge with ground truth
    diag = diag.rename(columns={"Subject": "PTID"})
    m = m.merge(diag, on=["PTID", "Phase"])
    
    return m


def clean_and_preprocess(df, overlap_test_set_path=None):
    """
    Clean and preprocess the merged clinical data.
    
    Args:
        df (pd.DataFrame): Merged clinical data.
        overlap_test_set_path (str): Path to overlap test set file.
        
    Returns:
        tuple: (X, y) processed features and labels.
    """
    print("Cleaning data...")
    # Drop unnecessary columns
    t = df.drop(["ID", "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2",
                 "update_stamp", "PTSOURCE", "PTDOBMM", "DX"], axis=1)
    
    # Fill missing values and remove columns with high percentage of missing data
    t = t.fillna(-4)
    t = t.replace("-4", -4)
    cols_to_delete = t.columns[(t == -4).sum() / len(t) > .70]
    t.drop(cols_to_delete, axis=1, inplace=True)
    
    # Clean and standardize work categories
    print("Processing categorical data...")
    t["PTWORK"] = t["PTWORK"].fillna("-4").astype(str).str.lower()
    t["PTWORK"] = t["PTWORK"].str.replace("housewife", "homemaker").str.replace("rn", "nurse").str.replace("bookeeper", "bookkeeper").str.replace("cpa", "accounting")
    
    # Standardize occupations
    work_mappings = {
        'teach': 'education',
        'bookkeep': 'bookkeeper',
        'wife': 'homemaker',
        'educat': 'education',
        'engineer': 'engineer',
        'eingineering': 'engineer',
        'computer programmer': 'engineer',
        'nurs': 'nurse',
        'manage': 'management',
        'therapist': 'therapist',
        'sales': 'sales',
        'admin': 'admin',
        'account': 'accounting',
        'real': 'real estate',
        'secretary': 'secretary',
        'professor': 'professor',
        'chem': 'chemist',
        'business': 'business',
        'writ': 'writing',
        'psych': 'psychology',
        'analys': 'analyst'
    }
    
    for pattern, replacement in work_mappings.items():
        t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*' + pattern + '.*$)', replacement, regex=True)
    
    # Group infrequent categories
    cond = t['PTWORK'].value_counts()
    threshold = 10
    t['PTWORK'] = np.where(t['PTWORK'].isin(cond.index[cond >= threshold]), t['PTWORK'], 'other')
    
    # Define categorical columns and quantitative columns
    categorical = ['PTGENDER', 'PTWORK', 'PTHOME', 'PTMARRY', 'PTEDUCAT', 'PTPLANG',
                   'NXVISUAL', 'PTNOTRT', 'NXTREMOR', 'NXAUDITO', 'PTHAND']
    
    # Check for additional categorical columns
    print("Identifying categorical variables...")
    cols_left = list(set(t.columns) - set(categorical) - set(["label", "Group", "GROUP", "Phase", "RID", "PTID", "PTRTYR", "EXAMDATE", "SUBJECT_KEY", "PTWRECNT"]))
    for col in cols_left:
        if len(t[col].value_counts()) < 10:
            categorical.append(col)
            
    # Remove specific columns
    t = t.drop(["PTRTYR", "EXAMDATE", "SUBJECT_KEY", "PTWRECNT"], axis=1)
    
    # Identify quantitative columns
    quant = list(set(cols_left) - set(categorical) - set(["label", "Group", "GROUP", "Phase", "RID", "PTID"]))
    
    # Apply one-hot encoding to categorical variables
    print("Applying one-hot encoding...")
    dfs = []
    for col in categorical:
        dfs.append(pd.get_dummies(t[col], prefix=col))
    
    cat = pd.concat(dfs, axis=1)
    
    # Combine categorical and quantitative features
    print("Combining features...")
    c = pd.concat([t[["PTID", "RID", "Phase", "Group"]].reset_index(), cat.reset_index(), t[quant].reset_index()], axis=1).drop("index", axis=1)
    
    # Group by PTID and take the most recent diagnosis
    c = c.groupby('PTID', group_keys=False).apply(lambda x: x.loc[x["Group"].astype(int).idxmax()]).drop("PTID", axis=1).reset_index()
    
    # Remove overlap test set if provided
    if overlap_test_set_path:
        print("Removing overlap test set...")
        ts = pd.read_csv(overlap_test_set_path).rename(columns={"subject": "PTID"})
        c = c[~c["PTID"].isin(list(ts["PTID"].values))]
    
    # Prepare features and labels
    print("Preparing features and labels...")
    cols = list(set(c.columns) - set(["PTID", "RID", "subject", "ID", "GROUP", "Group", "label", "Phase", "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2", "update_stamp", "DX_x", "DX_y", "Unnamed: 0"]))
    X = c[cols].values
    y = c["Group"].astype(int).values
    
    return X, y


def main(args):
    """
    Main function to run the clinical data preprocessing.
    
    Args:
        args: Command line arguments.
    """
    # Load data
    df = load_data(
        args.ground_truth, 
        args.demo, 
        args.neuro, 
        args.clinical, 
        args.dxsum
    )
    
    # Clean and preprocess
    X, y = clean_and_preprocess(df, args.overlap_test)
    
    # Split into train and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    
    # Save the processed data
    print("Saving processed data...")
    pd.DataFrame(X_train).to_pickle(args.output_dir + "/X_train_c.pkl")
    pd.DataFrame(y_train).to_pickle(args.output_dir + "/y_train_c.pkl")
    pd.DataFrame(X_test).to_pickle(args.output_dir + "/X_test_c.pkl")
    pd.DataFrame(y_test).to_pickle(args.output_dir + "/y_test_c.pkl")
    
    print(f"Clinical data preprocessing complete. Files saved to {args.output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess clinical data for Alzheimer's disease classification")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth diagnosis file")
    parser.add_argument("--demo", type=str, required=True, help="Path to demographic data file (PTDEMOG.csv)")
    parser.add_argument("--neuro", type=str, required=True, help="Path to neurological examination data file (NEUROEXM.csv)")
    parser.add_argument("--clinical", type=str, required=True, help="Path to clinical data file (ADSP_PHC_COGN.csv)")
    parser.add_argument("--dxsum", type=str, required=True, help="Path to diagnosis summary file (DXSUM_PDXCONV_ADNIALL.csv)")
    parser.add_argument("--overlap_test", type=str, default=None, help="Path to overlap test set file (optional)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save processed data")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split")
    
    args = parser.parse_args()
    main(args) 