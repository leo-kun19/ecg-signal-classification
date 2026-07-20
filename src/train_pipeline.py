import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import wfdb
import ast
from transformers import RobertaTokenizer, RobertaModel
import warnings
warnings.filterwarnings('ignore')

class ECGDataset(Dataset):
    """Dataset class for ECG signals and reports"""
    
    def __init__(self, ecg_signals, reports, labels, tokenizer=None, max_length=128):
        self.ecg_signals = ecg_signals
        self.reports = reports
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        ecg = torch.FloatTensor(self.ecg_signals[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        item = {
            'ecg': ecg,
            'label': label
        }
        
        if self.tokenizer is not None:
            report = str(self.reports[idx])
            encoding = self.tokenizer(
                report,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            item['input_ids'] = encoding['input_ids'].squeeze()
            item['attention_mask'] = encoding['attention_mask'].squeeze()
        
        return item

class ECGLSTMModel(nn.Module):
    """LSTM model for ECG classification"""
    
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(ECGLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class ECGRoBERTaModel(nn.Module):
    """RoBERTa model for ECG report classification"""
    
    def __init__(self, model_name='roberta-base', num_classes=2, dropout=0.3):
        super(ECGRoBERTaModel, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa doesn't have pooler_output, use last_hidden_state mean pooling
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        
        return output

def load_ecg_signals(df, data_path, max_samples=1000):
    """Load ECG signals from files"""
    print(f"Loading ECG signals from {max_samples} samples...")
    
    ecg_signals = []
    valid_indices = []
    
    for i, (idx, row) in enumerate(df.head(max_samples).iterrows()):
        try:
            filename = row['filename_lr']  # Use low resolution files
            file_path = Path(data_path) / filename
            
            if file_path.with_suffix('.dat').exists():
                record = wfdb.rdrecord(str(file_path))
                # Normalize the signal
                signal = (record.p_signal - np.mean(record.p_signal, axis=0)) / np.std(record.p_signal, axis=0)
                ecg_signals.append(signal)
                valid_indices.append(idx)
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{max_samples} signals...")
                
        except Exception as e:
            continue
    
    print(f"✓ Successfully loaded {len(ecg_signals)} ECG signals")
    return np.array(ecg_signals), valid_indices

def prepare_labels(df, valid_indices, target_codes=['NORM', 'ABQRS']):
    """Prepare binary labels from SCP codes"""
    print(f"Preparing labels for codes: {target_codes}")
    
    labels = []
    reports = []
    
    for idx in valid_indices:
        row = df.loc[idx]
        
        # Parse SCP codes
        try:
            scp_codes = ast.literal_eval(row['scp_codes'])
            
            # Binary classification: Normal vs Abnormal
            if any(code in scp_codes for code in target_codes[:1]):  # NORM
                label = 0  # Normal
            else:
                label = 1  # Abnormal
                
            labels.append(label)
            reports.append(row['clean_report'] if 'clean_report' in row else row['report'])
            
        except:
            # Default to abnormal if can't parse
            labels.append(1)
            reports.append(row['clean_report'] if 'clean_report' in row else row['report'])
    
    print(f"✓ Label distribution: Normal={labels.count(0)}, Abnormal={labels.count(1)}")
    return np.array(labels), reports

def train_lstm_model(ecg_signals, labels, num_epochs=10, batch_size=32):
    """Train LSTM model on ECG signals"""
    print(f"\n=== Training LSTM Model ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        ecg_signals, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ECGDataset(X_train, [""]*len(X_train), y_train)
    test_dataset = ECGDataset(X_test, [""]*len(X_test), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ECGLSTMModel(input_size=12, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            ecg = batch['ecg'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluation
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            ecg = batch['ecg'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(ecg)
            _, predicted = torch.max(outputs, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Print results
    print("\n=== LSTM Model Results ===")
    print("Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Normal', 'Abnormal']))
    
    return model

def train_roberta_model(reports, labels, num_epochs=3, batch_size=16):
    """Train RoBERTa model on ECG reports"""
    print(f"\n=== Training RoBERTa Model ===")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        reports, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ECGDataset(
        [np.zeros((1000, 12))]*len(X_train), X_train, y_train, 
        tokenizer=tokenizer, max_length=128
    )
    test_dataset = ECGDataset(
        [np.zeros((1000, 12))]*len(X_test), X_test, y_test, 
        tokenizer=tokenizer, max_length=128
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ECGRoBERTaModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluation
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Print results
    print("\n=== RoBERTa Model Results ===")
    print("Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Normal', 'Abnormal']))
    
    return model

def main():
    print("=" * 60)
    print("PTB-XL Training Pipeline")
    print("=" * 60)
    
    # Load preprocessed data
    data_path = './data/physionet.org/files/ptb-xl/1.0.3'
    preprocessed_path = './data/preprocessed_data.csv'
    
    if not Path(preprocessed_path).exists():
        print("❌ Preprocessed data not found. Please run preprocessing first.")
        return
    
    df = pd.read_csv(preprocessed_path)
    print(f"✓ Loaded preprocessed data: {len(df)} records")
    
    # Load ECG signals (subset for demo)
    ecg_signals, valid_indices = load_ecg_signals(df, data_path, max_samples=500)
    
    # Prepare labels and reports
    labels, reports = prepare_labels(df, valid_indices)
    
    # Train LSTM model on ECG signals
    lstm_model = train_lstm_model(ecg_signals, labels, num_epochs=5)
    
    # Train RoBERTa model on reports
    roberta_model = train_roberta_model(reports, labels, num_epochs=2)
    
    # Save models
    torch.save(lstm_model.state_dict(), './models/lstm_model.pth')
    torch.save(roberta_model.state_dict(), './models/roberta_model.pth')
    print("\n✓ Models saved to ./models/")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
