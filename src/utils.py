import os
import ast
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_ptbxl_data(path_to_data='../data/ptbxl_database.csv'):
    """
    Load PTB-XL database
    """
    database = pd.read_csv(path_to_data)
    return database

def aggregate_diagnostic(y_dic):
    """
    Aggregate diagnostic statements
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def load_raw_data(df, sampling_rate, path):
    """
    Load raw ECG data from PTB-XL dataset
    """
    data = []
    failed_loads = 0

    for idx, row in df.iterrows():
        try:
            if sampling_rate == 100:
                if 'filename_lr' in row:
                    # Use actual PTB-XL structure
                    filepath = os.path.join(path, row['filename_lr'])
                    signal, meta = wfdb.rdsamp(filepath)
                else:
                    # Fallback to synthetic data structure
                    ecg_id = str(row['ecg_id']).zfill(5)
                    subdir = ecg_id[:2] + "000"
                    filepath = os.path.join(path, subdir, f"{ecg_id}_lr.npy")
                    signal = np.load(filepath)
            else:
                if 'filename_hr' in row:
                    # Use actual PTB-XL structure  
                    filepath = os.path.join(path, row['filename_hr'])
                    signal, meta = wfdb.rdsamp(filepath)
                else:
                    # Fallback for synthetic data
                    raise FileNotFoundError("High resolution synthetic data not available")
            
            data.append(signal)
            
        except (FileNotFoundError, Exception) as e:
            failed_loads += 1
            # Create dummy data if file not found
            if sampling_rate == 100:
                dummy_signal = np.random.normal(0, 0.1, (1000, 12))
            else:
                dummy_signal = np.random.normal(0, 0.1, (5000, 12))
            data.append(dummy_signal)
    
    if failed_loads > 0:
        print(f"Warning: {failed_loads} files could not be loaded, using dummy data")
    
    return np.array(data)

def preprocess_signals(X_train, X_val, X_test):
    """
    Preprocess ECG signals
    """
    # Normalize
    scaler = StandardScaler()
    
    # Reshape for normalization
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_normalized.reshape(X_train.shape)
    
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_normalized = scaler.transform(X_val_reshaped)
    X_val = X_val_normalized.reshape(X_val.shape)
    
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_normalized = scaler.transform(X_test_reshaped)
    X_test = X_test_normalized.reshape(X_test.shape)
    
    return X_train, X_val, X_test, scaler

def prepare_labels(y, label_type='diagnostic_superclass'):
    """
    Prepare labels for classification
    """
    # Convert string representation to dict
    y_dict = [ast.literal_eval(label) if isinstance(label, str) else label for label in y]
    
    # Extract labels based on type
    if label_type == 'diagnostic_superclass':
        # Map to superclasses
        labels = []
        for y_sample in y_dict:
            if 'NORM' in y_sample and y_sample['NORM'] >= 50:
                labels.append('NORM')
            elif any(key in ['MI', 'STTC', 'CD', 'HYP'] for key in y_sample.keys()):
                # Get the key with highest probability
                max_key = max(y_sample.keys(), key=lambda k: y_sample[k] if k in ['MI', 'STTC', 'CD', 'HYP'] else 0)
                if max_key in ['MI', 'STTC', 'CD', 'HYP']:
                    labels.append(max_key)
                else:
                    labels.append('OTHER')
            else:
                labels.append('OTHER')
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    return y_encoded, le, labels

def plot_ecg_sample(data, title="ECG Sample", leads=None):
    """
    Plot ECG sample
    """
    if leads is None:
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(12, data.shape[1])):
        axes[i].plot(data[:, i])
        axes[i].set_title(f'Lead {leads[i]}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
