"""
Machine Learning Model for MongoDB Data (Scikit-learn version)
Compatible with Python 3.14+ - Uses Random Forest, XGBoost instead of CNN
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MongoDBMLPredictor:
    """
    Machine Learning predictor using scikit-learn models
    Works with Python 3.14+ (no TensorFlow required)
    """
    
    def __init__(self, mongo_uri=None, database_name='shopcam', collection_name='clients', task_type='classification'):
        """
        Initialize the ML Predictor
        
        Args:
            mongo_uri (str): MongoDB connection URI (reads from .env if None)
            database_name (str): Name of the database
            collection_name (str): Name of the collection
            task_type (str): 'classification' or 'regression'
        """
        # Load from environment if not provided
        if mongo_uri is None:
            mongo_uri = os.getenv('MONGO_DB_STRING')
            if not mongo_uri:
                raise ValueError("MongoDB URI not found. Set MONGO_DB_STRING in .env file")
        
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if task_type == 'classification' else None
        self.feature_names = []
        self.training_history = {}
        
    def connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            self.client.server_info()
            
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
            print(f"✓ Columns: {list(df.columns)}")
            
            return df
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_data(self, df, target_column, drop_columns=[]):
        """
        Preprocess data for ML training
        
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
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].mean())
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names before encoding
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables in features
        X = pd.get_dummies(X, drop_first=True)
        
        # Update feature names after encoding
        self.feature_names = X.columns.tolist()
        
        # Encode target if classification
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)
            print(f"✓ Target classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if self.task_type == 'classification' else None
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"✓ Training samples: {X_train.shape[0]}")
        print(f"✓ Testing samples: {X_test.shape[0]}")
        print(f"✓ Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, model_type='random_forest'):
        """
        Build ML model
        
        Args:
            model_type (str): 'random_forest', 'gradient_boosting', or 'xgboost'
            
        Returns:
            model: Initialized model
        """
        print(f"\n--- Building {model_type.upper()} Model ---")
        
        if self.task_type == 'classification':
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                except ImportError:
                    print("⚠ XGBoost not installed, using Random Forest instead")
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        else:  # regression
            if model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    self.model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                except ImportError:
                    print("⚠ XGBoost not installed, using Random Forest instead")
                    self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        print(f"✓ Model created: {type(self.model).__name__}")
        return self.model
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data (optional)
            
        Returns:
            model: Trained model
        """
        print("\n--- Training Model ---")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Training score
        train_score = self.model.score(X_train, y_train)
        print(f"✓ Training score: {train_score:.4f}")
        
        # Validation score if test data provided
        if X_test is not None and y_test is not None:
            test_score = self.model.score(X_test, y_test)
            print(f"✓ Test score: {test_score:.4f}")
            
            self.training_history = {
                'train_score': train_score,
                'test_score': test_score
            }
        
        # Cross-validation (only if enough data)
        if len(X_train) >= 3:
            cv_folds = min(3, len(X_train))
            try:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, n_jobs=-1)
                print(f"✓ Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                self.training_history['cv_scores'] = cv_scores
            except ValueError as e:
                print(f"⚠ Cross-validation skipped: {str(e)}")
        else:
            print("⚠ Cross-validation skipped due to insufficient data")
        
        print("\n✓ Training completed!")
        return self.model
    
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
        
        predictions = self.model.predict(X_test)
        
        if self.task_type == 'classification':
            accuracy = self.model.score(X_test, y_test)
            print(f"Test Accuracy: {accuracy:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, predictions, 
                                       target_names=self.label_encoder.classes_))
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'classification_report': classification_report(y_test, predictions, output_dict=True)
            }
        
        else:  # regression
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"Test MSE: {mse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test R² Score: {r2:.4f}")
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': predictions
            }
    
    def get_feature_importance(self):
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\n--- Top 10 Most Important Features ---")
            print(feature_importance_df.head(10))
            
            # Plot
            plt.figure(figsize=(10, 6))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance plot saved as 'feature_importance.png'")
            plt.show()
            
            return feature_importance_df
        else:
            print("Model doesn't support feature importance")
            return None
    
    def plot_results(self, y_test, predictions):
        """Plot model results"""
        if self.task_type == 'classification':
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("✓ Confusion matrix saved as 'confusion_matrix.png'")
            plt.show()
        
        else:  # regression
            # Prediction vs Actual
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual Values')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
            print("✓ Prediction plot saved as 'predictions_vs_actual.png'")
            plt.show()
    
    def save_model(self, model_path='ml_model.pkl'):
        """Save trained model and artifacts"""
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save label encoder if exists
        if self.label_encoder:
            joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__,
            'timestamp': datetime.now().isoformat(),
            'training_history': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in self.training_history.items()}
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to scaler.pkl")
        print(f"✓ Metadata saved to model_metadata.json")
    
    def load_model(self, model_path='ml_model.pkl'):
        """Load trained model and artifacts"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('scaler.pkl')
        
        if os.path.exists('label_encoder.pkl'):
            self.label_encoder = joblib.load('label_encoder.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            self.task_type = metadata['task_type']
        
        print(f"✓ Model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SHOPCAM ML PREDICTOR (Scikit-learn)")
    print("="*60)
    
    # Initialize predictor (will read from .env)
    predictor = MongoDBMLPredictor(
        database_name='shopcam',
        collection_name='clients',
        task_type='classification'
    )
    
    # Connect to MongoDB
    if predictor.connect_to_mongodb():
        # Fetch data
        df = predictor.fetch_data()
        
        if not df.empty:
            print("\nAvailable columns:", list(df.columns))
            print("\nPlease update TARGET_COLUMN in the script with your actual target column")
