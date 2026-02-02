# 📘 Ultra-Detailed Setup Guide - Shopcam ML Predictive System

**Complete Comprehensive Guide with Latest Fixes and Optimizations**

*Last Updated: February 2, 2026*

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [MongoDB Database Setup](#mongodb-database-setup)
5. [Installation & Dependencies](#installation--dependencies)
6. [Configuration Files](#configuration-files)
7. [Data Exploration](#data-exploration)
8. [Model Training](#model-training)
9. [Results Analysis](#results-analysis)
10. [Predictions & Inference](#predictions--inference)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Advanced Features](#advanced-features)
13. [Performance Optimization](#performance-optimization)
14. [Best Practices](#best-practices)
15. [FAQ](#faq)
16. [Complete Code Examples](#complete-code-examples)
17. [API Reference](#api-reference)

---

## 🔍 System Overview

This guide covers the complete setup and usage of a machine learning system that connects to MongoDB Atlas, extracts data from collections (clients, commandes, produits), and trains predictive models using scikit-learn algorithms.

### Key Features
- ✅ **Python 3.14 Compatible** (No TensorFlow required)
- ✅ **MongoDB Atlas Integration** with secure connection
- ✅ **Multiple ML Algorithms**: Random Forest, Gradient Boosting, XGBoost
- ✅ **Automatic Data Preprocessing** with missing value handling
- ✅ **Cross-Validation** optimized for small datasets
- ✅ **Model Persistence** with joblib
- ✅ **Visualization** with matplotlib/seaborn
- ✅ **Error Handling** and logging
- ✅ **PowerShell Scripts** for easy execution

### Recent Fixes Applied
- Fixed cross-validation errors for small datasets (≤10 samples)
- Resolved pandas ChainedAssignmentError warnings
- Optimized memory usage for plotting
- Improved error handling for MongoDB connections

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10+ (you have Windows ✅)
- **Python**: 3.8-3.14 (you have 3.14 ✅)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 500MB free space
- **Network**: Internet for MongoDB Atlas

### Your Current Environment
- **Python Path**: `C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe`
- **Working Directory**: `C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel`
- **MongoDB**: Atlas Cluster (shopcam database)

---

## 📁 Project Structure

```
Tabe-Joel/
├── 📄 app.py                          # Flask web app (optional)
├── 📄 ml_mongodb_model.py             # Main ML model class
├── 📄 train_shopcam_sklearn.py        # Training script
├── 📄 explore_shopcam_data.py         # Data exploration tool
├── 📄 predict.py                      # Prediction script
├── 📄 config.py                       # Configuration settings
├── 📄 requirements.txt                # Python dependencies
├── 📄 .env                            # Environment variables (MongoDB credentials)
├── 📄 DETAILED_SETUP_GUIDE.md         # This guide
├── 📄 run_training.ps1               # PowerShell training script
├── 📄 run_explore.ps1                # PowerShell exploration script
├── 📁 __pycache__/                    # Python cache files
└── 📁 Generated files after training:
    ├── 📄 clients_ml_model.pkl        # Trained model
    ├── 📄 scaler.pkl                  # Feature scaler
    ├── 📄 label_encoder.pkl           # Label encoder (classification)
    ├── 📄 model_metadata.json         # Model metadata
    ├── 📄 confusion_matrix.png        # Confusion matrix plot
    └── 📄 feature_importance.png      # Feature importance plot
```

---

## 🗄️ MongoDB Database Setup

### Your Database Structure
- **Cluster**: Cluster0 (MongoDB Atlas)
- **Database**: shopcam
- **Collections**:
  - `clients`: 10 documents (customer data)
  - `commandes`: 15 documents (order data)
  - `produits`: 8 documents (product data)

### Sample Data Structure

**clients collection:**
```json
{
  "_id": ObjectId("..."),
  "nom": "John Doe",
  "ville": "Douala",
  "age": 30
}
```

**commandes collection:**
```json
{
  "_id": ObjectId("..."),
  "client_id": "...",
  "produit_id": "...",
  "quantite": 5,
  "date_commande": "2024-01-15"
}
```

**produits collection:**
```json
{
  "_id": ObjectId("..."),
  "nom": "Product A",
  "prix": 100.0,
  "categorie": "Electronics"
}
```

---

## 📦 Installation & Dependencies

### Step 1: Install Python Packages

Run this command in PowerShell:

```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" -m pip install -r requirements.txt
```

### requirements.txt Content:
```
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
pymongo==4.6.0
matplotlib==3.8.0
seaborn==0.13.0
joblib==1.3.0
python-dotenv==1.0.0
xgboost==2.0.0
flask==3.0.0
```

### Step 2: Verify Installation

```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" -c "import sklearn, pandas, pymongo; print('All packages installed successfully')"
```

---

## ⚙️ Configuration Files

### .env File Setup

Create a `.env` file in your project directory:

```env
# MongoDB Atlas Configuration
MONGODB_URI=mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=shopcam

# Optional: Model Configuration
MODEL_TYPE=random_forest
TASK_TYPE=classification
TARGET_COLUMN=ville
```

**Security Note**: Never commit `.env` files to version control.

### config.py Content:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Settings
    MONGODB_URI = os.getenv('MONGODB_URI')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'shopcam')
    
    # Model Settings
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'random_forest')
    TASK_TYPE = os.getenv('TASK_TYPE', 'classification')
    TARGET_COLUMN = os.getenv('TARGET_COLUMN', 'ville')
    
    # Collections
    COLLECTIONS = {
        'clients': {'target': 'ville', 'task': 'classification'},
        'commandes': {'target': 'quantite', 'task': 'regression'},
        'produits': {'target': 'prix', 'task': 'regression'}
    }
```

---

## 🔍 Data Exploration

### Step 1: Run Data Exploration

```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" explore_shopcam_data.py
```

### What You'll See:

```
============================================================
SHOPCAM DATA EXPLORATION TOOL
============================================================

Available Collections:
1. clients   (10 documents)
2. commandes (15 documents)
3. produits  (8 documents)

Select collection to explore (1-3): 1

============================================================
EXPLORING CLIENTS COLLECTION
============================================================

✓ Connected to MongoDB: shopcam.clients
✓ Fetched 10 records

--- Column Information ---
Column: nom (object)
  - Unique values: 10
  - Sample values: ['John Doe', 'Jane Smith', ...]

Column: ville (object)
  - Unique values: 2
  - Sample values: ['Douala', 'Yaoundé']

Column: age (int64)
  - Range: 25-45
  - Mean: 32.5

--- Data Sample ---
   nom    ville  age
0  John   Douala   30
1  Jane  Yaoundé   25
...
```

### Step 2: Understand Your Data

**Key Insights:**
- Small datasets (8-15 samples per collection)
- Limited features available
- Some categorical variables need encoding
- Missing values handled automatically

---

## 🤖 Model Training

### Step 1: Run Training Script

```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" train_shopcam_sklearn.py
```

### Training Process:

1. **MongoDB Connection**: Connects to your Atlas cluster
2. **Data Fetching**: Retrieves all documents from selected collection
3. **Preprocessing**: Handles missing values, encodes categorical data
4. **Train/Test Split**: 80/20 split with stratification
5. **Model Building**: Creates Random Forest model
6. **Training**: Fits model with cross-validation
7. **Evaluation**: Tests on holdout set
8. **Visualization**: Generates plots and saves model

### Current Training Results (Latest Run):

```
✓ Training score: 1.0000
✓ Test score: 0.0000
✓ Cross-validation score: 0.3889 (+/- 0.0786)
```

**Analysis:**
- **Overfitting**: Training score 1.0 vs Test score 0.0
- **Small Data Issue**: Only 2 test samples
- **CV Score**: More realistic 38.89% accuracy

---

## 📊 Results Analysis

### Understanding Metrics

#### Classification Metrics:
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

#### Your Results:
```
Test Accuracy: 0.0000

Classification Report:
              precision    recall  f1-score   support
      Douala       0.00      0.00      0.00       1.0
     Yaoundé       0.00      0.00      0.00       1.0
```

**Why Poor Performance?**
1. **Tiny Test Set**: Only 2 samples
2. **Class Imbalance**: Equal classes but small numbers
3. **Limited Features**: Only 'age' feature after preprocessing
4. **Random Chance**: 50% baseline for 2 classes

### Generated Files:

After training, you'll have:
- `clients_ml_model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- `label_encoder.pkl`: For categorical targets
- `model_metadata.json`: Training information
- `confusion_matrix.png`: Prediction visualization
- `feature_importance.png`: Feature rankings

---

## 🔮 Predictions & Inference

### Using Saved Models

Create a prediction script:

```python
import joblib
import pandas as pd
from ml_mongodb_model import MongoDBMLPredictor

# Load model
model = joblib.load('clients_ml_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Create predictor instance
predictor = MongoDBMLPredictor()
predictor.model = model
predictor.scaler = scaler
predictor.label_encoder = label_encoder

# New data for prediction
new_data = pd.DataFrame({
    'nom': ['New Client'],
    'age': [35]
})

# Make prediction
prediction = predictor.predict(new_data)
print(f"Predicted city: {prediction}")
```

### Batch Predictions

```python
# Load multiple samples
batch_data = pd.DataFrame({
    'nom': ['Client A', 'Client B', 'Client C'],
    'age': [28, 42, 31]
})

predictions = predictor.predict(batch_data)
for i, pred in enumerate(predictions):
    print(f"Client {i+1}: {pred}")
```

---

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Cross-Validation Errors
**Error:** `ValueError: n_splits=5 cannot be greater than the number of members in each class`

**Solution:** ✅ **Fixed** - Code now uses max 3 folds and skips CV for very small datasets.

#### 2. ChainedAssignmentError
**Error:** `A value is being set on a copy of a DataFrame`

**Solution:** ✅ **Fixed** - Replaced `inplace=True` with proper assignments.

#### 3. Memory Allocation Errors
**Error:** `unable to alloc XXXXX bytes`

**Solution:** ✅ **Fixed** - Optimized plotting and memory usage.

#### 4. MongoDB Connection Issues
**Error:** `pymongo.errors.ServerSelectionTimeoutError`

**Solutions:**
- Check internet connection
- Verify MongoDB URI in `.env`
- Ensure IP whitelist includes your IP
- Check cluster status in Atlas dashboard

#### 5. Import Errors
**Error:** `ModuleNotFoundError: No module named 'pymongo'`

**Solution:**
```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" -m pip install pymongo
```

#### 6. Python Path Issues
**Error:** `python: command not found`

**Solution:** Use full path:
```powershell
& "C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe" script.py
```

#### 7. Small Dataset Warnings
**Warning:** Poor model performance

**Solution:** Add more data to MongoDB collections (target: 100+ samples per collection)

#### 8. Plot Display Issues
**Issue:** Plots don't show in terminal

**Solution:** Plots are saved as PNG files automatically. Check project directory for `.png` files.

---

## 🚀 Advanced Features

### Custom Model Configuration

```python
from ml_mongodb_model import MongoDBMLPredictor

# Advanced configuration
predictor = MongoDBMLPredictor(
    model_type='gradient_boosting',
    task_type='classification',
    target_column='ville',
    test_size=0.3,
    random_state=42
)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(
    predictor.model, 
    param_grid, 
    cv=3, 
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

### Model Comparison

```python
models_to_test = ['random_forest', 'gradient_boosting', 'xgboost']

for model_type in models_to_test:
    predictor = MongoDBMLPredictor(model_type=model_type)
    predictor.connect_to_mongodb()
    data = predictor.fetch_data('clients')
    X_train, X_test, y_train, y_test = predictor.preprocess_data(data, 'ville')
    predictor.build_model()
    predictor.train(X_train, y_train, X_test, y_test)
    print(f"{model_type}: Test score = {predictor.training_history['test_score']:.4f}")
```

### Web API with Flask

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('clients_ml_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Preprocess
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## ⚡ Performance Optimization

### For Small Datasets (< 50 samples):

1. **Use Simple Models**: Random Forest over complex neural networks
2. **Reduce CV Folds**: Max 3 folds instead of 5
3. **Feature Selection**: Limit to most important features
4. **Avoid Overfitting**: Monitor train/test score gap

### For Larger Datasets:

1. **Increase CV Folds**: Up to 5 or 10
2. **Use Advanced Models**: XGBoost, neural networks
3. **Hyperparameter Tuning**: Grid/random search
4. **Feature Engineering**: Create new features

### Memory Optimization:

1. **Batch Processing**: Process data in chunks
2. **Garbage Collection**: Explicit cleanup
3. **Efficient Data Types**: Use appropriate dtypes
4. **Model Serialization**: Save/load models efficiently

---

## 📚 Best Practices

### Data Management
- **Regular Backups**: Backup MongoDB data regularly
- **Data Validation**: Validate data before insertion
- **Incremental Updates**: Update models as new data arrives
- **Version Control**: Track model versions

### Model Development
- **Cross-Validation**: Always use CV for reliable metrics
- **Train/Test Split**: Never evaluate on training data
- **Feature Importance**: Analyze which features matter
- **Error Analysis**: Study misclassifications

### Production Deployment
- **Model Monitoring**: Track performance over time
- **Retraining**: Schedule periodic model updates
- **Fallback Plans**: Have backup models/predictions
- **Logging**: Log all predictions and errors

### Code Quality
- **Error Handling**: Catch and handle exceptions
- **Documentation**: Comment complex logic
- **Modular Code**: Separate concerns
- **Testing**: Unit tests for critical functions

---

## ❓ FAQ

### Q: Why is test accuracy 0.0?
**A:** With only 2 test samples and random chance at 50%, it's possible to get 0 correct predictions. Add more data for reliable evaluation.

### Q: Can I use this with TensorFlow?
**A:** The current implementation uses scikit-learn for Python 3.14 compatibility. TensorFlow support can be added for Python < 3.14.

### Q: How do I add more collections?
**A:** Update `config.py` with new collection names and target columns, then modify `train_shopcam_sklearn.py`.

### Q: What if my data has different column names?
**A:** Update the `TARGET_COLUMN` in the training functions or config file.

### Q: Can I run this on Linux/Mac?
**A:** Yes, update the Python path and PowerShell commands to bash equivalents.

### Q: How do I improve model performance?
**A:** 1) Add more data, 2) Feature engineering, 3) Hyperparameter tuning, 4) Try different algorithms.

### Q: Is the model saved securely?
**A:** Models are saved locally. For production, consider encryption and access controls.

### Q: Can I use this for real-time predictions?
**A:** Yes, load the saved model and use it for predictions without retraining.

### Q: What if MongoDB connection fails?
**A:** Check your `.env` file, network connection, and MongoDB Atlas settings.

### Q: How do I update the model with new data?
**A:** Retrain the model with updated data from MongoDB.

---

## 💻 Complete Code Examples

### Full Training Script Example

```python
import os
from ml_mongodb_model import MongoDBMLPredictor

def train_clients_model():
    # Initialize predictor
    predictor = MongoDBMLPredictor(
        model_type='random_forest',
        task_type='classification',
        target_column='ville'
    )
    
    # Connect to MongoDB
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    # Fetch data
    data = predictor.fetch_data('clients')
    print(f"Fetched {len(data)} records")
    
    # Preprocess
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        data, 'ville', test_size=0.2
    )
    
    # Build and train model
    predictor.build_model()
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    predictor.evaluate(X_test, y_test)
    
    # Plot results
    predictions = predictor.model.predict(X_test)
    predictor.plot_results(y_test, predictions)
    
    # Save model
    predictor.save_model('clients_ml_model.pkl')
    
    print("Training complete!")

if __name__ == '__main__':
    train_clients_model()
```

### Data Exploration Script

```python
from ml_mongodb_model import MongoDBMLPredictor
import pandas as pd

def explore_collection(collection_name):
    predictor = MongoDBMLPredictor()
    
    if not predictor.connect_to_mongodb():
        return
    
    # Fetch data
    data = predictor.fetch_data(collection_name)
    
    print(f"\n=== {collection_name.upper()} COLLECTION ===")
    print(f"Records: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    
    # Data types
    print("\n--- Data Types ---")
    print(data.dtypes)
    
    # Statistics
    print("\n--- Statistics ---")
    print(data.describe(include='all'))
    
    # Sample data
    print("\n--- Sample Data ---")
    print(data.head())
    
    # Missing values
    print("\n--- Missing Values ---")
    print(data.isnull().sum())

if __name__ == '__main__':
    collections = ['clients', 'commandes', 'produits']
    for col in collections:
        explore_collection(col)
```

---

## 📖 API Reference

### MongoDBMLPredictor Class

#### Constructor
```python
MongoDBMLPredictor(
    mongo_uri=None,           # MongoDB connection string
    database_name='shopcam',  # Database name
    model_type='random_forest', # Model algorithm
    task_type='classification', # 'classification' or 'regression'
    target_column=None,       # Target column name
    test_size=0.2,            # Train/test split ratio
    random_state=42           # Random seed
)
```

#### Methods

**connect_to_mongodb()**
- Returns: bool (connection success)

**fetch_data(collection_name)**
- Parameters: collection_name (str)
- Returns: pandas.DataFrame

**preprocess_data(data, target_column, test_size=0.2)**
- Parameters: data (DataFrame), target_column (str), test_size (float)
- Returns: X_train, X_test, y_train, y_test

**build_model()**
- Creates and configures the ML model

**train(X_train, y_train, X_test=None, y_test=None)**
- Parameters: training data and optional test data
- Returns: trained model

**evaluate(X_test, y_test)**
- Parameters: test data
- Returns: evaluation metrics dict

**predict(X)**
- Parameters: feature data
- Returns: predictions

**plot_results(y_test, predictions)**
- Saves visualization plots

**save_model(filepath)**
- Parameters: save path
- Saves model, scaler, encoder, metadata

---

## 🎯 Quick Start Checklist

- [ ] Verify Python 3.14 installation
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `.env` file with MongoDB credentials
- [ ] Run data exploration: `python explore_shopcam_data.py`
- [ ] Train first model: `python train_shopcam_sklearn.py`
- [ ] Check generated files and plots
- [ ] Make test predictions
- [ ] Review troubleshooting section if issues arise

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the troubleshooting section
2. Review error messages carefully
3. Verify your environment matches requirements
4. Ensure MongoDB Atlas is accessible
5. Check that all dependencies are installed

**Remember**: With small datasets (8-15 samples), expect limited model performance. Focus on adding more data for better results.

---

*This guide was generated on February 2, 2026, and covers all aspects of the Shopcam ML system including recent fixes and optimizations.*</content>
<parameter name="filePath">c:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel\ULTRA_DETAILED_SETUP_GUIDE.md