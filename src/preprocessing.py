import pandas as pd
import numpy as np
from pathlib import Path
import wfdb
import warnings
warnings.filterwarnings('ignore')

# Preprocessing Functions

def preprocess_ecg_data(df, data_path):
    """
    Function to preprocess ECG data files.
    """
    print(f"\n=== ECG DATA PREPROCESSING ===")
    print(f"Data path: {data_path}")
    
    # Check if data path exists
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        return None
    
    # Sample ECG file from first record
    sample_filename = df['filename_lr'].iloc[0]  # e.g., 'records100/00000/00001_lr'
    sample_file_path = Path(data_path) / sample_filename
    
    print(f"Sample filename from CSV: {sample_filename}")
    print(f"Full sample path: {sample_file_path}")
    
    # Check if sample file exists (without extension)
    if sample_file_path.with_suffix('.dat').exists():
        print(f"✓ Sample ECG file found: {sample_file_path}.dat")
        
        # Try to read the sample ECG
        try:
            record = wfdb.rdrecord(str(sample_file_path))
            print(f"✓ Successfully read ECG record")
            print(f"  - Sampling frequency: {record.fs} Hz")
            print(f"  - Number of leads: {record.n_sig}")
            print(f"  - Signal length: {record.sig_len} samples")
            print(f"  - Duration: {record.sig_len / record.fs:.2f} seconds")
            
            return record
        except Exception as e:
            print(f"❌ Error reading ECG file: {e}")
            return None
    else:
        print(f"❌ Sample ECG file not found: {sample_file_path}.dat")
        return None


def preprocess_reports(df):
    """
    Function to preprocess ECG reports text.
    """
    print(f"\n=== REPORT PREPROCESSING ===")
    
    # Basic statistics
    total_reports = len(df)
    non_empty_reports = df['report'].notna().sum()
    print(f"Total reports: {total_reports}")
    print(f"Non-empty reports: {non_empty_reports}")
    
    if non_empty_reports > 0:
        # Clean reports
        df['clean_report'] = df['report'].apply(lambda x: str(x).lower().replace('\n', ' ').strip() if pd.notna(x) else '')
        
        # Calculate lengths
        df['report_length'] = df['clean_report'].apply(lambda x: len(x.split()) if x else 0)
        
        # Sample reports
        print(f"\nSample cleaned reports:")
        sample_reports = df[df['clean_report'] != ''].head(3)
        for i, (idx, row) in enumerate(sample_reports.iterrows()):
            print(f"  {i+1}. [{row['report_length']} words]: {row['clean_report'][:100]}...")
        
        print(f"✓ Reports preprocessing complete")
        print(f"  - Average report length: {df['report_length'].mean():.1f} words")
        print(f"  - Median report length: {df['report_length'].median():.1f} words")
        
        return df
    else:
        print("❌ No reports to preprocess")
        return df


def analyze_scp_codes(df):
    """
    Analyze and preprocess SCP diagnostic codes
    """
    print(f"\n=== SCP CODES PREPROCESSING ===")
    
    if 'scp_codes' not in df.columns:
        print("❌ SCP codes column not found")
        return df
    
    # Parse SCP codes and create binary labels
    import ast
    
    # Get all unique SCP codes
    all_scp_codes = set()
    valid_rows = 0
    
    for codes in df['scp_codes']:
        try:
            if pd.notna(codes):
                scp_dict = ast.literal_eval(codes)
                all_scp_codes.update(scp_dict.keys())
                valid_rows += 1
        except:
            continue
    
    print(f"Valid SCP code rows: {valid_rows}/{len(df)}")
    print(f"Total unique SCP codes: {len(all_scp_codes)}")
    print(f"Most common codes: {sorted(list(all_scp_codes))[:10]}")
    
    # For now, just keep the original format
    print("✓ SCP codes analysis complete")
    
    return df

def main():
    print("=" * 60)
    print("PTB-XL Dataset Preprocessing")
    print("=" * 60)
    
    # Locate the dataset file
    dataset_path = './data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv'
    ecg_data_path = './data/physionet.org/files/ptb-xl/1.0.3'

    if not Path(dataset_path).exists():
        print(f"❌ Database file not found at {dataset_path}")
        return

    print(f"✓ Found database at: {dataset_path}")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"✓ Loaded dataset: {len(df)} records, {len(df.columns)} columns")

    # Preprocess ECG Data
    sample_record = preprocess_ecg_data(df, ecg_data_path)

    # Preprocess Reports
    df = preprocess_reports(df)
    
    # Analyze SCP codes
    df = analyze_scp_codes(df)

    # Save preprocessed data
    output_path = './data/preprocessed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Preprocessed data saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
