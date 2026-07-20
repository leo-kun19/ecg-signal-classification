import os
from src.utils import load_ptbxl_data, load_raw_data, preprocess_signals, prepare_labels
from src.lstm_model import ECGLSTMModel
from src.bert_model import ECGBERTModel, preprocess_reports
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Setup the data
    data_path = './data/ptbxl_sample.csv'
    df = load_ptbxl_data(data_path)

    # Preprocess ECG signals
    sampling_rate = 100  # This is just for demonstration!
    signals = load_raw_data(df, sampling_rate=sampling_rate, path='./data/records100_sample/')
    
    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = train_test_split(signals, df['scp_codes'].values, test_size=0.2, stratify=df['scp_codes'].values, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    # Preprocess signals
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test)
    
    # Prepare labels
    y_train_encoded, le, y_train_labels = prepare_labels(y_train)
    y_val_encoded, _, _ = prepare_labels(y_val)
    y_test_encoded, _, _ = prepare_labels(y_test)
    num_classes = len(le.classes_)

    # Train LSTM model
    lstm_model = ECGLSTMModel(input_shape=X_train.shape[1:], num_classes=num_classes)
    lstm_model.train(X_train, y_train_encoded, X_val, y_val_encoded, epochs=10, batch_size=32)

    # Evaluate LSTM model
    loss, accuracy = lstm_model.evaluate(X_test, y_test_encoded)
    print(f'LSTM Model Test loss: {loss}, Test accuracy: {accuracy}')

    # Preprocess text reports
    reports = preprocess_reports(df['report'])
    
    # Train BERT model
    bert_model = ECGBERTModel()
    train_loader, val_loader = bert_model.prepare_data(reports, y_train_encoded)
    bert_model.train(train_loader, val_loader, epochs=3)

    # Evaluate BERT model
    val_accuracy = bert_model.evaluate(val_loader)
    print(f'BERT Model Validation Accuracy: {val_accuracy}')

