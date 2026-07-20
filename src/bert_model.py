import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm

class ECGReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ECGBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(ECGBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        return output

class ECGRoBERTaModel:
    def __init__(self, model_name='roberta-base', num_classes=5, max_length=512, dropout_rate=0.3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ECGBERTClassifier(model_name, num_classes, dropout_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using model: {model_name}")
        print(f"Device: {self.device}")
        
    def prepare_data(self, texts, labels, batch_size=16, test_size=0.2):
        """
        Prepare data for training
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ECGReportDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = ECGReportDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=5, learning_rate=2e-5):
        """
        Train the BERT model
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc='Training'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_accuracy = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation accuracy: {val_accuracy:.4f}')
            print('-' * 50)
        
        return train_losses, val_accuracies
    
    def evaluate(self, data_loader):
        """
        Evaluate the model
        """
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts
        """
        dataset = ECGReportDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
        
        return predictions
    
    def save_model(self, path):
        """
        Save the trained model
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        Load a trained model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

def preprocess_reports(reports):
    """
    Preprocess ECG reports for BERT
    """
    # Clean and normalize text
    processed_reports = []
    for report in reports:
        if pd.isna(report):
            processed_reports.append("")
        else:
            # Convert to lowercase and clean
            clean_report = str(report).lower().strip()
            processed_reports.append(clean_report)
    
    return processed_reports
