"""
Prediction Script for CNN MongoDB Model
Use trained model to make predictions on new data
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json
from datetime import datetime
import argparse
import sys


class CNNPredictor:
    """
    Load trained CNN model and make predictions
    """
    
    def __init__(self, model_path='cnn_model.h5'):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = []
        
    def load_artifacts(self):
        """Load all model artifacts"""
        try:
            # Load model
            print(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
            print("✓ Model loaded")
            
            # Load scaler
            self.scaler = joblib.load('scaler.pkl')
            print("✓ Scaler loaded")
            
            # Load label encoder if exists
            try:
                self.label_encoder = joblib.load('label_encoder.pkl')
                print("✓ Label encoder loaded")
            except FileNotFoundError:
                print("⚠ No label encoder found (regression task)")
            
            # Load metadata
            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata.get('feature_names', [])
            task_type = self.metadata.get('task_type', 'unknown')
            
            print(f"✓ Metadata loaded")
            print(f"  Task Type: {task_type}")
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Trained: {self.metadata.get('timestamp', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model artifacts: {str(e)}")
            return False
    
    def preprocess_input(self, data):
        """
        Preprocess input data
        
        Args:
            data (pd.DataFrame or dict): Input data
            
        Returns:
            array: Preprocessed data ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            # Check if single sample or multiple samples
            if all(isinstance(v, (list, np.ndarray)) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict or DataFrame")
        
        # Handle categorical encoding (one-hot)
        df = pd.get_dummies(df, drop_first=True)
        
        # Ensure all features from training are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Remove extra features and reorder
        df = df[self.feature_names]
        
        # Scale features
        X = self.scaler.transform(df)
        
        # Reshape for CNN
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X
    
    def predict(self, data, return_probabilities=False):
        """
        Make predictions on input data
        
        Args:
            data: Input data (DataFrame or dict)
            return_probabilities (bool): Return class probabilities for classification
            
        Returns:
            array or dict: Predictions
        """
        if self.model is None:
            print("✗ Model not loaded. Call load_artifacts() first.")
            return None
        
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Process based on task type
        task_type = self.metadata.get('task_type', 'classification')
        
        if task_type == 'classification':
            pred_classes = np.argmax(predictions, axis=1)
            
            if self.label_encoder:
                pred_labels = self.label_encoder.inverse_transform(pred_classes)
            else:
                pred_labels = pred_classes
            
            if return_probabilities:
                return {
                    'predictions': pred_labels,
                    'probabilities': predictions,
                    'class_indices': pred_classes
                }
            else:
                return pred_labels
        
        else:  # regression
            return predictions.flatten()
    
    def predict_single(self, sample_data):
        """
        Predict for a single sample
        
        Args:
            sample_data (dict): Single sample data
            
        Returns:
            Prediction result
        """
        result = self.predict(sample_data, return_probabilities=True)
        
        task_type = self.metadata.get('task_type', 'classification')
        
        if task_type == 'classification':
            print("\n=== Prediction Result ===")
            print(f"Predicted Class: {result['predictions'][0]}")
            print(f"\nClass Probabilities:")
            
            if self.label_encoder:
                classes = self.label_encoder.classes_
                probs = result['probabilities'][0]
                for cls, prob in zip(classes, probs):
                    print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
            
            return result['predictions'][0]
        
        else:  # regression
            prediction = result[0]
            print("\n=== Prediction Result ===")
            print(f"Predicted Value: {prediction:.4f}")
            return prediction
    
    def predict_batch(self, data):
        """
        Predict for multiple samples
        
        Args:
            data (pd.DataFrame or list of dicts): Batch data
            
        Returns:
            pd.DataFrame: Predictions with input data
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        predictions = self.predict(df)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        print(f"\n✓ Made {len(predictions)} predictions")
        print(f"\nSample predictions:")
        print(df.head(10))
        
        return df
    
    def save_predictions(self, data, output_path='predictions.csv'):
        """
        Make predictions and save to file
        
        Args:
            data: Input data
            output_path (str): Output file path
        """
        df_predictions = self.predict_batch(data)
        df_predictions.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")
    
    def get_feature_importance(self):
        """
        Estimate feature importance (simple method)
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        print("\n=== Feature Importance Analysis ===")
        print("Note: This is a simplified analysis for CNNs")
        
        # Get first layer weights
        first_layer_weights = self.model.layers[0].get_weights()[0]
        
        # Calculate average absolute weight per feature
        importance = np.mean(np.abs(first_layer_weights), axis=(1, 2))
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return importance_df


def main():
    """Command-line interface for predictions"""
    parser = argparse.ArgumentParser(description='CNN Model Prediction Tool')
    
    parser.add_argument('--model', type=str, default='cnn_model.h5',
                       help='Path to trained model')
    
    parser.add_argument('--input', type=str,
                       help='Path to input CSV file')
    
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to output predictions file')
    
    parser.add_argument('--single', action='store_true',
                       help='Interactive single prediction mode')
    
    parser.add_argument('--importance', action='store_true',
                       help='Show feature importance')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CNNPredictor(model_path=args.model)
    
    # Load model artifacts
    if not predictor.load_artifacts():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Feature importance
    if args.importance:
        predictor.get_feature_importance()
        return
    
    # Single prediction mode
    if args.single:
        print("\n=== Interactive Prediction Mode ===")
        print("Enter feature values (or 'quit' to exit):\n")
        
        sample = {}
        for feature in predictor.feature_names:
            value = input(f"{feature}: ")
            if value.lower() == 'quit':
                return
            
            # Try to convert to number
            try:
                sample[feature] = float(value)
            except ValueError:
                sample[feature] = value
        
        predictor.predict_single(sample)
        return
    
    # Batch prediction mode
    if args.input:
        print(f"\n=== Batch Prediction Mode ===")
        print(f"Reading data from {args.input}...")
        
        try:
            data = pd.read_csv(args.input)
            predictor.save_predictions(data, args.output)
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            sys.exit(1)
    else:
        print("Please provide --input file or use --single mode")
        parser.print_help()


if __name__ == "__main__":
    # If no command-line arguments, run example
    if len(sys.argv) == 1:
        print("="*60)
        print("CNN MODEL PREDICTION TOOL")
        print("="*60)
        print("\nUsage Examples:")
        print("\n1. Batch predictions from CSV:")
        print("   python predict.py --input data.csv --output predictions.csv")
        print("\n2. Interactive single prediction:")
        print("   python predict.py --single")
        print("\n3. Show feature importance:")
        print("   python predict.py --importance")
        print("\n4. Use custom model:")
        print("   python predict.py --model my_model.h5 --input data.csv")
        print("\n" + "="*60)
        
        # Example programmatic usage
        print("\n=== Example: Programmatic Usage ===\n")
        
        example_code = '''
# Initialize predictor
predictor = CNNPredictor(model_path='cnn_model.h5')

# Load model
predictor.load_artifacts()

# Single prediction
sample = {
    'feature1': 10.5,
    'feature2': 20.3,
    'feature3': 5.8
}
prediction = predictor.predict_single(sample)

# Batch prediction
data = pd.read_csv('new_data.csv')
predictions = predictor.predict_batch(data)

# Save predictions
predictor.save_predictions(data, 'output.csv')
        '''
        
        print(example_code)
    else:
        main()
