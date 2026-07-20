import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class ECGLSTMModel:
    def __init__(self, input_shape, num_classes, lstm_units=128, dropout_rate=0.3):
        """
        Initialize LSTM model for ECG classification
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """
        Build LSTM model architecture
        """
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        # LSTM layers
        model.add(LSTM(self.lstm_units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        model.add(LSTM(self.lstm_units//2))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        self.model = model
        return model
    
    def get_callbacks(self, model_save_path):
        """
        Get training callbacks
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_save_path='../models/lstm_model.h5'):
        """
        Train the LSTM model
        """
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks(model_save_path)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X_test)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()

def create_simple_lstm_model(input_shape, num_classes):
    """
    Create a simpler LSTM model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
