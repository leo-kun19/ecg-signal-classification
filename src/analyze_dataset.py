import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_ptbxl_dataset():
    """
    Comprehensive analysis of PTB-XL dataset
    """
    print("=" * 60)
    print("PTB-XL Dataset Analysis")
    print("=" * 60)
    
    # Load dataset
    try:
        # Try the downloaded path first
        if Path('./data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv').exists():
            df = pd.read_csv('./data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv')
        else:
            df = pd.read_csv('./data/ptbxl_database.csv')
        print(f"✓ Dataset loaded successfully: {len(df)} records")
    except FileNotFoundError:
        print("❌ PTB-XL database not found. Please run download_data.py first")
        return
    
    # Basic info
    print(f"\n1. BASIC INFORMATION")
    print(f"   - Total records: {len(df)}")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    # Column information
    print(f"\n2. COLUMN INFORMATION")
    for i, col in enumerate(df.columns):
        print(f"   {i+1:2d}. {col}")
    
    # Demographic analysis
    analyze_demographics(df)
    
    # SCP codes analysis
    analyze_scp_codes(df)
    
    # Report analysis
    analyze_reports(df)
    
    # ECG signal analysis
    analyze_ecg_signals(df)
    
    # Save analysis results
    save_analysis_results(df)

def analyze_demographics(df):
    """
    Analyze demographic information
    """
    print(f"\n3. DEMOGRAPHIC ANALYSIS")
    
    # Age analysis
    if 'age' in df.columns:
        age_stats = df['age'].describe()
        print(f"   Age statistics:")
        print(f"   - Mean: {age_stats['mean']:.1f} years")
        print(f"   - Median: {age_stats['50%']:.1f} years")
        print(f"   - Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
        print(f"   - Missing values: {df['age'].isnull().sum()}")
    
    # Sex analysis
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        print(f"   Sex distribution:")
        for sex, count in sex_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {sex}: {count} ({percentage:.1f}%)")
    
    # Height and Weight
    if 'height' in df.columns:
        height_stats = df['height'].describe()
        print(f"   Height: mean={height_stats['mean']:.1f}cm, std={height_stats['std']:.1f}cm")
    
    if 'weight' in df.columns:
        weight_stats = df['weight'].describe()
        print(f"   Weight: mean={weight_stats['mean']:.1f}kg, std={weight_stats['std']:.1f}kg")

def analyze_scp_codes(df):
    """
    Analyze SCP diagnostic codes
    """
    print(f"\n4. SCP CODES ANALYSIS")
    
    if 'scp_codes' not in df.columns:
        print("   ❌ SCP codes column not found")
        return
    
    # Parse SCP codes
    all_scp_codes = []
    for codes in df['scp_codes']:
        try:
            if pd.notna(codes):
                scp_dict = ast.literal_eval(codes)
                all_scp_codes.extend(list(scp_dict.keys()))
        except:
            continue
    
    # Count occurrences
    scp_counter = Counter(all_scp_codes)
    
    print(f"   - Total unique SCP codes: {len(scp_counter)}")
    print(f"   - Most common SCP codes:")
    for code, count in scp_counter.most_common(10):
        percentage = (count / len(df)) * 100
        print(f"     {code}: {count} ({percentage:.1f}%)")

def analyze_reports(df):
    """
    Analyze ECG reports for text classification
    """
    print(f"\n5. REPORT ANALYSIS")
    
    if 'report' not in df.columns:
        print("   ❌ Report column not found")
        return
    
    # Basic report statistics
    non_empty_reports = df['report'].notna().sum()
    empty_reports = df['report'].isna().sum()
    
    print(f"   - Non-empty reports: {non_empty_reports} ({(non_empty_reports/len(df)*100):.1f}%)")
    print(f"   - Empty reports: {empty_reports} ({(empty_reports/len(df)*100):.1f}%)")
    
    # Report length analysis
    if non_empty_reports > 0:
        report_lengths = df['report'].dropna().apply(lambda x: len(str(x).split()))
        print(f"   - Average report length: {report_lengths.mean():.1f} words")
        print(f"   - Median report length: {report_lengths.median():.1f} words")
        print(f"   - Report length range: {report_lengths.min()} - {report_lengths.max()} words")
        
        # Sample reports
        print(f"   - Sample reports:")
        sample_reports = df['report'].dropna().sample(3, random_state=42)
        for i, report in enumerate(sample_reports):
            print(f"     {i+1}. {str(report)[:100]}...")

def analyze_ecg_signals(df):
    """
    Analyze ECG signals information
    """
    print(f"\n6. ECG SIGNALS ANALYSIS")
    
    # Check filename columns
    has_lr = 'filename_lr' in df.columns
    has_hr = 'filename_hr' in df.columns
    
    print(f"   - Low resolution files (100Hz): {'✓' if has_lr else '❌'}")
    print(f"   - High resolution files (500Hz): {'✓' if has_hr else '❌'}")
    
    if has_lr:
        print(f"   - Sample LR filename: {df['filename_lr'].iloc[0]}")
    if has_hr:
        print(f"   - Sample HR filename: {df['filename_hr'].iloc[0]}")
    
    # Recording information
    if 'recording_date' in df.columns:
        dates = pd.to_datetime(df['recording_date'], errors='coerce')
        print(f"   - Recording date range: {dates.min()} to {dates.max()}")
    
    if 'device' in df.columns:
        devices = df['device'].value_counts()
        print(f"   - Recording devices: {list(devices.index)}")

def save_analysis_results(df):
    """
    Save analysis results for preprocessing
    """
    print(f"\n7. SAVING ANALYSIS RESULTS")
    
    # Create analysis summary
    analysis_summary = {
        'total_records': len(df),
        'columns': list(df.columns),
        'has_reports': 'report' in df.columns,
        'has_scp_codes': 'scp_codes' in df.columns,
        'has_lr_files': 'filename_lr' in df.columns,
        'has_hr_files': 'filename_hr' in df.columns,
    }
    
    # Add demographic stats
    if 'age' in df.columns:
        analysis_summary['age_stats'] = df['age'].describe().to_dict()
    
    if 'sex' in df.columns:
        analysis_summary['sex_distribution'] = df['sex'].value_counts().to_dict()
    
    # Add report stats
    if 'report' in df.columns:
        analysis_summary['report_stats'] = {
            'non_empty': df['report'].notna().sum(),
            'empty': df['report'].isna().sum(),
        }
        
        if df['report'].notna().sum() > 0:
            report_lengths = df['report'].dropna().apply(lambda x: len(str(x).split()))
            analysis_summary['report_length_stats'] = report_lengths.describe().to_dict()
    
    # Save to file
    import json
    with open('./results/dataset_analysis.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    
    print("   ✓ Analysis results saved to ./results/dataset_analysis.json")

def create_visualizations(df):
    """
    Create visualizations for the dataset
    """
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PTB-XL Dataset Analysis', fontsize=16)
    
    # Age distribution
    if 'age' in df.columns:
        axes[0, 0].hist(df['age'].dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age (years)')
        axes[0, 0].set_ylabel('Frequency')
    
    # Sex distribution
    if 'sex' in df.columns:
        sex_counts = df['sex'].value_counts()
        axes[0, 1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Sex Distribution')
    
    # Report length distribution
    if 'report' in df.columns:
        report_lengths = df['report'].dropna().apply(lambda x: len(str(x).split()))
        axes[1, 0].hist(report_lengths, bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Report Length Distribution')
        axes[1, 0].set_xlabel('Words')
        axes[1, 0].set_ylabel('Frequency')
    
    # Missing values heatmap
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        axes[1, 1].bar(range(len(missing_data)), missing_data.values)
        axes[1, 1].set_title('Missing Values by Column')
        axes[1, 1].set_xlabel('Columns')
        axes[1, 1].set_ylabel('Missing Count')
        axes[1, 1].set_xticks(range(len(missing_data)))
        axes[1, 1].set_xticklabels(missing_data.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('./results/dataset_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display issues
    
    print("   ✓ Visualizations saved to ./results/dataset_visualization.png")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('./results', exist_ok=True)
    
    # Run analysis
    analyze_ptbxl_dataset()
    
    # Create visualizations
    try:
        # Try the downloaded path first
        if Path('./data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv').exists():
            df = pd.read_csv('./data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv')
        else:
            df = pd.read_csv('./data/ptbxl_database.csv')
        create_visualizations(df)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
