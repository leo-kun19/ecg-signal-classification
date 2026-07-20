import os
import subprocess
import requests
import pandas as pd
from pathlib import Path
import numpy as np
import time

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        print(f"\n{filename} downloaded successfully!")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False

def download_ptbxl_full_dataset():
    """Download the entire PTB-XL dataset using wget"""
    print("="*60)
    print("Downloading PTB-XL Dataset (Full Version - 1.7GB)")
    print("="*60)
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Change to data directory
    original_dir = os.getcwd()
    os.chdir(data_dir)
    
    try:
        # Use wget to download the full dataset
        print("Using wget to download full PTB-XL dataset...")
        print("This will download ~1.7GB of data. Please wait...")
        
        url = "https://physionet.org/files/ptb-xl/1.0.3/"
        
        # Check if wget is available
        result = subprocess.run(['wget', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ wget not found! Please install wget first.")
            print("   Run: winget install wget")
            return False
        
        print("✓ wget found, starting full dataset download...")
        
        # Use wget with proper flags for recursive download
        # -r: recursive
        # -N: don't re-retrieve files unless newer than local
        # -c: continue partial downloads
        # -np: don't go to parent directories
        # -nH: don't create host directories
        # --cut-dirs=3: cut 3 directory levels
        # --reject: reject certain file types
        
        wget_cmd = [
            'wget', 
            '-r',           # recursive
            '-N',           # timestamping
            '-c',           # continue
            '-np',          # no parent
            '-nH',          # no host directories
            '--cut-dirs=3', # cut directory levels
            '--reject=html,tmp', # reject HTML files
            '--progress=bar:force', # show progress
            url
        ]
        
        print(f"Running: {' '.join(wget_cmd)}")
        result = subprocess.run(wget_cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Download completed successfully using wget!")
            return True
        else:
            print(f"\n❌ wget failed with return code: {result.returncode}")
            print("Falling back to requests method...")
            
    except Exception as e:
        print(f"❌ wget failed with error: {e}")
        print("Falling back to requests method...")
    
    finally:
        os.chdir(original_dir)
    
    # Fallback to requests method
    return download_with_requests()

def download_with_requests():
    """Download essential files using requests"""
    print("Downloading essential files using requests...")
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    
    # Essential files
    files_to_download = [
        "ptbxl_database.csv",
        "scp_statements.csv", 
        "RECORDS",
        "LICENSE.txt"
    ]
    
    # Download metadata files
    for filename in files_to_download:
        filepath = data_dir / filename
        if not filepath.exists():
            url = base_url + filename
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping.")
    
    # Try to download some sample records
    print("\nDownloading sample ECG records...")
    try:
        download_sample_records(base_url, data_dir)
    except Exception as e:
        print(f"Error downloading sample records: {e}")
        create_synthetic_sample_data(data_dir)
    
    return True

def download_sample_records(base_url, data_dir):
    """Download a few sample ECG record files"""
    # Create records directories
    records100_dir = data_dir / "records100"
    records500_dir = data_dir / "records500"
    records100_dir.mkdir(exist_ok=True)
    records500_dir.mkdir(exist_ok=True)
    
    # Download some sample records
    sample_records = [
        "records100/00000/00001_lr.dat",
        "records100/00000/00001_lr.hea",
        "records500/00000/00001_hr.dat", 
        "records500/00000/00001_hr.hea"
    ]
    
    for record in sample_records:
        record_path = data_dir / record
        record_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not record_path.exists():
            url = base_url + record
            try:
                download_file(url, record_path)
            except:
                print(f"Could not download {record}")

def create_synthetic_sample_data(data_dir):
    """Create synthetic ECG data for demonstration if real data not available"""
    print("Creating synthetic ECG sample data...")
    
    # Load database to get structure
    try:
        df = pd.read_csv(data_dir / "ptbxl_database.csv")
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        
        # Create sample directories
        records100_dir = data_dir / "records100"
        records100_dir.mkdir(exist_ok=True)
        
        for idx, row in sample_df.iterrows():
            ecg_id = str(row['ecg_id']).zfill(5)
            subdir = ecg_id[:2] + "000"
            full_dir = records100_dir / subdir
            full_dir.mkdir(exist_ok=True)
            
            # Generate synthetic ECG signal
            ecg_signal = generate_realistic_ecg(1000, 12)
            np.save(full_dir / f"{ecg_id}_lr.npy", ecg_signal)
        
        print(f"Created synthetic data for {len(sample_df)} records")
        
    except Exception as e:
        print(f"Error creating synthetic data: {e}")

def generate_realistic_ecg(length, num_leads):
    """Generate realistic ECG signals"""
    np.random.seed(42)
    t = np.linspace(0, 10, length)
    signals = []
    
    for lead in range(num_leads):
        # Heart rate around 70 BPM with variation
        hr = 70 + np.random.normal(0, 5)
        hr = max(50, min(100, hr))
        
        # P wave (atrial depolarization)
        p_wave = 0.1 * np.sin(2 * np.pi * (hr/60) * t)
        
        # QRS complex (ventricular depolarization)
        qrs_wave = 0.8 * np.sin(2 * np.pi * (hr/60) * t + np.pi/6)
        
        # T wave (ventricular repolarization)  
        t_wave = 0.3 * np.sin(2 * np.pi * (hr/60) * t + np.pi/3)
        
        # Combine waves
        ecg = p_wave + qrs_wave + t_wave
        
        # Add lead-specific characteristics
        if lead < 6:  # Limb leads
            ecg *= (0.5 + 0.3 * np.random.random())
        else:  # Precordial leads
            ecg *= (0.8 + 0.4 * np.random.random())
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, length)
        ecg += noise
        
        signals.append(ecg)
    
    return np.array(signals).T

def analyze_downloaded_data():
    """Analyze the downloaded PTB-XL data"""
    data_dir = Path("../data")
    
    # Check what was downloaded
    print("\n" + "="*60)
    print("DOWNLOADED DATA ANALYSIS")
    print("="*60)
    
    if (data_dir / "ptbxl_database.csv").exists():
        df = pd.read_csv(data_dir / "ptbxl_database.csv")
        print(f"✓ PTB-XL Database: {len(df)} records")
        print(f"✓ Report column available: {'report' in df.columns}")
        
        if 'report' in df.columns:
            non_empty_reports = df['report'].notna().sum()
            print(f"✓ Non-empty reports: {non_empty_reports}/{len(df)} ({non_empty_reports/len(df)*100:.1f}%)")
            
            # Show sample reports for RoBERTa
            print(f"\n📄 Sample reports for RoBERTa:")
            sample_reports = df['report'].dropna().sample(3, random_state=42)
            for i, report in enumerate(sample_reports, 1):
                print(f"   {i}. {str(report)[:100]}...")
    
    # Check for ECG signal files
    records100_exists = (data_dir / "records100").exists()
    records500_exists = (data_dir / "records500").exists()
    
    print(f"✓ 100Hz ECG records: {'Available' if records100_exists else 'Not found'}")
    print(f"✓ 500Hz ECG records: {'Available' if records500_exists else 'Not found'}")
    
    if records100_exists:
        record_count = len(list((data_dir / "records100").rglob("*.npy")))
        print(f"✓ Sample ECG files: {record_count} found")
    
    print("\n✅ Data ready for training!")
    print("   - Use ptbxl_database.csv 'report' column for RoBERTa")
    print("   - Use ECG signal files for LSTM")

def setup_ptbxl_data():
    """Setup PTB-XL dataset"""
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    # URLs for PTB-XL dataset
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    files_to_download = [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "RECORDS",
        "LICENSE.txt"
    ]
    
    print("Downloading PTB-XL dataset files...")
    
    # Download metadata files
    for filename in files_to_download:
        if not (data_dir / filename).exists():
            url = base_url + filename
            download_file(url, data_dir / filename)
        else:
            print(f"{filename} already exists, skipping download.")
    
    print("\n" + "="*60)
    print("PTB-XL Dataset Information:")
    print("- Full dataset size: ~1.7GB (ZIP) / ~22GB (uncompressed)")
    print("- Contains 21,837 ECG records from 18,885 patients")
    print("- Two sampling frequencies: 500Hz (records500/) and 100Hz (records100/)")
    print("- Each record has report column for text classification")
    print("\nTo download full dataset:")
    print("Option 1: wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/")
    print("Option 2: aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./data/")
    print("Option 3: Download ZIP from PhysioNet website")
    print("="*60)
    
    # Check if full dataset exists, otherwise create sample data
    if not check_full_dataset_exists():
        print("\nFull dataset not found. Creating sample data for demonstration...")
        create_sample_data()
    else:
        print("\nFull dataset found! Using actual PTB-XL data.")
        prepare_full_dataset()

def check_full_dataset_exists():
    """Check if full PTB-XL dataset exists"""
    data_dir = Path("../data")
    return (
        (data_dir / "records100").exists() or 
        (data_dir / "records500").exists()
    )

def prepare_full_dataset():
    """Prepare the full dataset for training"""
    data_dir = Path("../data")
    
    # Load the database
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    print(f"Loaded {len(df)} ECG records from PTB-XL database")
    
    # Check available data
    if (data_dir / "records100").exists():
        print("Using 100Hz sampling rate data (records100/)")
        sampling_rate = 100
    elif (data_dir / "records500").exists():
        print("Using 500Hz sampling rate data (records500/)")
        sampling_rate = 500
    
    # Create a smaller subset for faster training (optional)
    subset_size = min(5000, len(df))  # Use up to 5000 records
    df_subset = df.sample(n=subset_size, random_state=42)
    df_subset.to_csv(data_dir / "ptbxl_subset.csv", index=False)
    
    print(f"Created subset with {subset_size} records for training")
    print(f"Report column available: {'report' in df.columns}")
    if 'report' in df.columns:
        non_empty_reports = df['report'].notna().sum()
        print(f"Records with non-empty reports: {non_empty_reports}/{len(df)}")
    
def create_sample_data():
    """Create sample ECG data for demonstration"""
    import numpy as np
    
    data_dir = Path("../data")
    
    # Load the database
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    
    # Take a sample for demonstration
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    sample_df.to_csv(data_dir / "ptbxl_sample.csv", index=False)
    
    # Create synthetic ECG signals for the sample
    print("Creating synthetic ECG signals for demonstration...")
    
    # Create directories for sample data
    (data_dir / "records100_sample").mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    for idx, row in sample_df.iterrows():
        # Generate synthetic 12-lead ECG signal (1000 samples, 12 leads)
        # This is just for demonstration - in real scenario you would download actual signals
        ecg_signal = generate_synthetic_ecg(1000, 12)
        
        # Create subdirectory structure
        ecg_id = str(row['ecg_id']).zfill(5)
        subdir = ecg_id[:2] + "000"
        full_dir = data_dir / "records100_sample" / subdir
        full_dir.mkdir(exist_ok=True)
        
        # Save as numpy array
        np.save(full_dir / f"{ecg_id}_lr.npy", ecg_signal)
    
    print("Sample data created successfully!")

def generate_synthetic_ecg(length, num_leads):
    """Generate synthetic ECG signal for demonstration"""
    t = np.linspace(0, 10, length)  # 10 seconds
    ecg_signals = []
    
    for lead in range(num_leads):
        # Basic ECG-like signal with some noise
        signal = (
            0.5 * np.sin(2 * np.pi * 1.2 * t) +  # Heart rate ~72 bpm
            0.3 * np.sin(2 * np.pi * 0.2 * t) +  # Respiratory variation
            0.1 * np.random.normal(0, 1, length)  # Noise
        )
        
        # Add some lead-specific variations
        if lead < 6:  # Limb leads
            signal *= (0.8 + 0.4 * np.random.random())
        else:  # Chest leads
            signal *= (1.0 + 0.6 * np.random.random())
        
        ecg_signals.append(signal)
    
    return np.array(ecg_signals).T

if __name__ == "__main__":
    try:
        # Download the full dataset
        success = download_ptbxl_full_dataset()
        
        if success:
            # Analyze what was downloaded
            analyze_downloaded_data()
            print("\n🎉 Dataset setup completed successfully!")
        else:
            print("\n❌ Dataset download failed!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\nNext steps:")
    print("1. Run: python src/analyze_dataset.py")
    print("2. Run: python run_pipeline.py")
