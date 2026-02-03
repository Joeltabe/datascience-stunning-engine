"""
CNN Model for MongoDB Data Training and Prediction
This script connects to MongoDB, trains a CNN model, and provides predictive insights.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

class MongoDBCNNPredictor:
    """
    A CNN-based predictive model that trains on MongoDB data.
    Supports both classification and regression tasks.
    """
    
    def __init__(self, mongo_uri, database_name, collection_name, task_type='classification'):
        """
        Initialize the MongoDB CNN Predictor
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database_name (str): Name of the database
            collection_name (str): Name of the collection
            task_type (str): 'classification' or 'regression'
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        self.history = None
        self.feature_names = []
        
    def connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            print(f"✓ Connected to MongoDB: {self.database_name}.{self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ Error connecting to MongoDB: {str(e)}")
            return False
    
    def fetch_data(self, query={}, limit=None):
        """
        Fetch data from MongoDB
        
        Args:
            query (dict): MongoDB query filter
            limit (int): Maximum number of documents to fetch
            
        Returns:
            pd.DataFrame: Fetched data as DataFrame
        """
        try:
            cursor = self.collection.find(query)
            if limit:
                cursor = cursor.limit(limit)
            
            data = list(cursor)
            df = pd.DataFrame(data)
            
            # Remove MongoDB _id field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            print(f"✓ Fetched {len(df)} records from MongoDB")
            return df
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_data(self, df, target_column, drop_columns=[]):
        """
        Preprocess data for CNN training
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target variable
            drop_columns (list): Columns to drop
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n--- Data Preprocessing ---")
        
        # Drop unnecessary columns
        df = df.drop(columns=drop_columns, errors='ignore')
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables in features
        X = pd.get_dummies(X, drop_first=True)
        
        # Encode target if classification
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)
            print(f"✓ Target classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Reshape for CNN (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"✓ Training samples: {X_train.shape[0]}")
        print(f"✓ Testing samples: {X_test.shape[0]}")
        print(f"✓ Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_cnn_model(self, input_shape, num_classes=None):
        """
        Build CNN architecture
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of classes for classification
            
        Returns:
            keras.Model: Compiled CNN model
        """
        model = Sequential([
            # First convolutional block
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Second convolutional block
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Third convolutional block
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3)
        ])
        
        # Output layer based on task type
        if self.task_type == 'classification':
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:  # regression
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        print("\n--- CNN Model Architecture ---")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the CNN model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.History: Training history
        """
        print("\n--- Training CNN Model ---")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_cnn_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("\n✓ Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n--- Model Evaluation ---")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        if self.task_type == 'classification':
            print(f"Test Loss: {results[0]:.4f}")
            print(f"Test Accuracy: {results[1]:.4f}")
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            return {
                'loss': results[0],
                'accuracy': results[1],
                'predictions': y_pred_classes
            }
        else:  # regression
            print(f"Test MSE: {results[0]:.4f}")
            print(f"Test MAE: {results[1]:.4f}")
            
            y_pred = self.model.predict(X_test).flatten()
            
            return {
                'mse': results[0],
                'mae': results[1],
                'predictions': y_pred
            }
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data: Input data (DataFrame or array)
            
        Returns:
            array: Predictions
        """
        if isinstance(data, pd.DataFrame):
            data = pd.get_dummies(data, drop_first=True)
            data = self.scaler.transform(data)
        
        data = data.reshape(data.shape[0], data.shape[1], 1)
        predictions = self.model.predict(data)
        
        if self.task_type == 'classification':
            pred_classes = np.argmax(predictions, axis=1)
            pred_labels = self.label_encoder.inverse_transform(pred_classes)
            return pred_labels
        else:
            return predictions.flatten()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metric plot
        metric_name = 'accuracy' if self.task_type == 'classification' else 'mae'
        axes[1].plot(self.history.history[metric_name], label=f'Training {metric_name}')
        axes[1].plot(self.history.history[f'val_{metric_name}'], label=f'Validation {metric_name}')
        axes[1].set_title(f'Model {metric_name.upper()}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name.upper())
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history plot saved as 'training_history.png'")
        plt.show()
    
    def save_model(self, model_path='cnn_model.h5', scaler_path='scaler.pkl'):
        """Save trained model and scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='cnn_model.h5', scaler_path='scaler.pkl'):
        """Load trained model and scaler"""
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if os.path.exists('label_encoder.pkl'):
            self.label_encoder = joblib.load('label_encoder.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            self.task_type = metadata['task_type']
        
        print(f"✓ Model loaded from {model_path}")
    
    def get_insights(self, X_test, y_test):
        """
        Generate predictive insights
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Insights and statistics
        """
        print("\n--- Generating Predictive Insights ---")
        
        predictions = self.model.predict(X_test)
        
        if self.task_type == 'classification':
            pred_classes = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import classification_report, confusion_matrix
            
            print("\nClassification Report:")
            print(classification_report(y_test, pred_classes))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, pred_classes)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("✓ Confusion matrix saved as 'confusion_matrix.png'")
            plt.show()
            
            insights = {
                'classification_report': classification_report(y_test, pred_classes, output_dict=True),
                'confusion_matrix': cm.tolist()
            }
        else:  # regression
            predictions = predictions.flatten()
            
            from sklearn.metrics import r2_score, mean_absolute_percentage_error
            
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            print(f"\nR² Score: {r2:.4f}")
            print(f"MAPE: {mape:.4f}")
            
            # Prediction vs Actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual Values')
            plt.grid(True)
            plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
            print("✓ Prediction plot saved as 'predictions_vs_actual.png'")
            plt.show()
            
            insights = {
                'r2_score': r2,
                'mape': mape,
                'predictions_sample': predictions[:10].tolist(),
                'actual_sample': y_test[:10].tolist()
            }
        
        return insights


# Example usage
if __name__ == "__main__":
    """
    Example: Train CNN on MongoDB data
    
    Prerequisites:
    1. MongoDB instance running
    2. Database and collection with data
    3. Update connection parameters below
    """
    
    # Configuration
    MONGO_URI = "mongodb://localhost:27017/"  # Update with your MongoDB URI
    DATABASE_NAME = "your_database"  # Update with your database name
    COLLECTION_NAME = "your_collection"  # Update with your collection name
    TARGET_COLUMN = "target"  # Update with your target column name
    TASK_TYPE = "classification"  # or "regression"
    
    # Initialize predictor
    predictor = MongoDBCNNPredictor(
        mongo_uri=MONGO_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
        task_type=TASK_TYPE
    )
    
    # Connect to MongoDB
    if predictor.connect_to_mongodb():
        # Fetch data
        df = predictor.fetch_data(limit=10000)
        
        if not df.empty:
            # Preprocess data
            X_train, X_test, y_train, y_test = predictor.preprocess_data(
                df, 
                target_column=TARGET_COLUMN,
                drop_columns=[]  # Add columns to drop if any
            )
            
            # Build model
            num_classes = len(predictor.label_encoder.classes_) if TASK_TYPE == 'classification' else None
            predictor.build_cnn_model(
                input_shape=(X_train.shape[1], 1),
                num_classes=num_classes
            )
            
            # Train model
            predictor.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
            
            # Evaluate model
            results = predictor.evaluate(X_test, y_test)
            
            # Generate insights
            insights = predictor.get_insights(X_test, y_test)
            
            # Plot training history
            predictor.plot_training_history()
            
            # Save model
            predictor.save_model()
            
            print("\n✓ CNN training and prediction pipeline completed successfully!")
