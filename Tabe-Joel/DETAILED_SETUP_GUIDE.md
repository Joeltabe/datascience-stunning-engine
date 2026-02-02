# 📘 Complete Setup Guide - Shopcam ML Predictive System

**A Comprehensive Step-by-Step Guide for Training Machine Learning Models on MongoDB Data**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Initial Setup](#initial-setup)
3. [Understanding Your Project Structure](#understanding-your-project-structure)
4. [MongoDB Database Overview](#mongodb-database-overview)
5. [Installation Guide](#installation-guide)
6. [Configuration](#configuration)
7. [Data Exploration](#data-exploration)
8. [Training Your First Model](#training-your-first-model)
9. [Understanding the Results](#understanding-the-results)
10. [Making Predictions](#making-predictions)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Usage](#advanced-usage)
13. [Best Practices](#best-practices)
14. [FAQ](#faq)

---

## 🖥️ System Requirements

### Minimum Requirements
- **Operating System**: Windows 10 or higher
- **Python Version**: 3.8 - 3.14 (You have Python 3.14 ✅)
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB free space
- **Internet Connection**: Required for MongoDB Atlas access

### Your Current Setup ✅
- **Python**: 3.14 (Located at `C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe`)
- **Operating System**: Windows
- **MongoDB**: Atlas Cloud (Cluster0)
- **Database**: shopcam

---

## 🚀 Initial Setup

### Step 1: Verify Python Installation

Open PowerShell and run:

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe --version
```

**Expected Output:**
```
Python 3.14.x
```

If you see this, Python is correctly installed ✅

### Step 2: Verify pip (Python Package Manager)

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip --version
```

**Expected Output:**
```
pip 25.3 or higher
```

### Step 3: Update pip (Optional but Recommended)

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip install --upgrade pip
```

---

## 📁 Understanding Your Project Structure

### Current Directory Structure

```
Tabe-Joel/
│
├── 📄 .env                              # MongoDB credentials (ALREADY CONFIGURED ✅)
│
├── 🐍 Python Scripts
│   ├── ml_mongodb_model.py              # Main ML model class
│   ├── train_shopcam_sklearn.py         # Training script (READY TO USE)
│   ├── explore_shopcam_data.py          # Data exploration tool
│   ├── cnn_mongodb_model.py             # CNN version (requires TensorFlow)
│   ├── data_preprocessing.py            # Data preprocessing utilities
│   ├── predict.py                       # Prediction script
│   └── config.py                        # Configuration settings
│
├── 📜 PowerShell Scripts (EASIEST TO USE)
│   ├── run_training.ps1                 # Double-click to train!
│   └── run_explore.ps1                  # Double-click to explore data!
│
├── 📖 Documentation
│   ├── START_HERE.md                    # Quick start guide
│   ├── DATA_STRUCTURE.md                # Your database summary
│   ├── SHOPCAM_QUICKSTART.md           # Quick reference
│   ├── README.md                        # General documentation
│   └── DETAILED_SETUP_GUIDE.md         # This file!
│
├── 📦 Dependencies
│   └── requirements.txt                 # Python packages (INSTALLED ✅)
│
└── 📊 Generated Files (Created after training)
    ├── clients_ml_model.pkl            # Trained model for clients
    ├── commandes_ml_model.pkl          # Trained model for commandes
    ├── produits_ml_model.pkl           # Trained model for produits
    ├── scaler.pkl                       # Feature scaler
    ├── label_encoder.pkl                # Label encoder (for classification)
    ├── model_metadata.json              # Model information
    ├── feature_importance.png           # Feature importance chart
    ├── confusion_matrix.png             # Classification results
    └── predictions_vs_actual.png        # Regression results
```

---

## 🗄️ MongoDB Database Overview

### Your MongoDB Atlas Connection

**Connection String:**
```
mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority
```

**Database Name:** `shopcam`

**Cluster:** `Cluster0`

### Collections in Your Database

#### 1️⃣ Collection: `clients` (10 documents)

**Purpose**: Store customer information

**Structure:**
| Column | Type    | Description           | Example Values          |
|--------|---------|----------------------|-------------------------|
| _id    | Integer | Unique client ID     | 1, 2, 3, 4, 5...       |
| nom    | String  | Client name          | Client_1, Client_2...  |
| ville  | String  | Client's city        | Douala, Yaoundé        |
| age    | Integer | Client's age         | 21, 22, 23, 24, 25...  |

**Sample Data:**
```
_id | nom       | ville   | age
----|-----------|---------|-----
1   | Client_1  | Douala  | 21
2   | Client_2  | Yaoundé | 22
3   | Client_3  | Douala  | 23
```

**Prediction Opportunities:**
- ✅ **Classification**: Predict `ville` (city) based on `age`
- ✅ **Regression**: Predict `age` based on other features
- ✅ **Use Case**: Customer segmentation, location prediction

---

#### 2️⃣ Collection: `commandes` (15 documents)

**Purpose**: Store order information

**Structure:**
| Column          | Type    | Description           | Example Values     |
|----------------|---------|----------------------|-------------------|
| _id            | Integer | Unique order ID      | 501, 502, 503...  |
| client_id      | Integer | Customer who ordered | 2, 3, 4, 5...     |
| produit_id     | Integer | Product ordered      | 102, 103, 104...  |
| quantite       | Integer | Quantity ordered     | 1, 2, 3           |
| ville_livraison| String  | Delivery city        | Douala, Yaoundé   |

**Sample Data:**
```
_id | client_id | produit_id | quantite | ville_livraison
----|-----------|------------|----------|----------------
501 | 2         | 102        | 2        | Douala
502 | 3         | 103        | 3        | Douala
503 | 4         | 104        | 1        | Yaoundé
```

**Prediction Opportunities:**
- ✅ **Regression**: Predict `quantite` (order quantity)
- ✅ **Classification**: Predict `ville_livraison` (delivery location)
- ✅ **Classification**: Predict `client_id` (identify customer patterns)
- ✅ **Use Case**: Demand forecasting, delivery optimization

---

#### 3️⃣ Collection: `produits` (8 documents)

**Purpose**: Store product catalog

**Structure:**
| Column | Type    | Description        | Example Values              |
|--------|---------|-------------------|----------------------------|
| _id    | Integer | Unique product ID | 101, 102, 103...           |
| nom    | String  | Product name      | Laptop HP, Smartphone...   |
| prix   | Integer | Price in CFA      | 15000, 150000, 350000...   |

**Sample Data:**
```
_id | nom        | prix
----|------------|--------
101 | Laptop HP  | 350000
102 | Smartphone | 150000
103 | Tablette   | 250000
104 | Ecran      | 80000
105 | Clavier    | 15000
```

**Prediction Opportunities:**
- ✅ **Regression**: Predict `prix` (product price)
- ✅ **Classification**: Classify product category
- ✅ **Use Case**: Price optimization, product recommendation

---

## 💾 Installation Guide

### Step 1: Navigate to Project Directory

Open PowerShell and navigate to your project folder:

```powershell
cd 'C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel'
```

**Verify you're in the correct directory:**
```powershell
pwd
```

**Expected Output:**
```
Path
----
C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel
```

### Step 2: Check Installed Packages

The required packages are already installed ✅, but you can verify:

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip list
```

**You should see:**
- ✅ scikit-learn (1.8.0 or higher)
- ✅ pandas (3.0.0 or higher)
- ✅ numpy (2.4.2 or higher)
- ✅ pymongo (4.16.0 or higher)
- ✅ matplotlib (3.10.8 or higher)
- ✅ seaborn (0.13.2 or higher)
- ✅ joblib (1.5.3 or higher)
- ✅ scipy (1.17.0 or higher)
- ✅ python-dotenv (1.2.1 or higher)

### Step 3: Verify Installation

Test if packages import correctly:

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -c "import sklearn, pandas, pymongo; print('All packages installed successfully!')"
```

**Expected Output:**
```
All packages installed successfully!
```

If you see this message, everything is ready! ✅

---

## ⚙️ Configuration

### Understanding the .env File

The `.env` file stores your MongoDB credentials securely.

**Location:** `C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel\.env`

**Current Contents:**
```dotenv
NGROK_AUTH_TOKEN = 38tqmWIWfCWtpoy3h8GzdyQRULA_DuY4PuGduyqnYTc3zGVV
MONGO_DB_STRING = mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority
```

**⚠️ Security Note:**
- Never share this file publicly
- Never commit to GitHub
- Keep credentials private

### How It Works

When you run the scripts, they automatically:
1. Load the `.env` file using `python-dotenv`
2. Read `MONGO_DB_STRING` variable
3. Connect to your MongoDB Atlas database
4. No need to hardcode credentials in your code!

**Example in code:**
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file
mongo_uri = os.getenv('MONGO_DB_STRING')  # Gets connection string
```

---

## 🔍 Data Exploration

### Method 1: Using PowerShell Script (EASIEST)

Simply double-click `run_explore.ps1` in Windows Explorer, or run:

```powershell
.\run_explore.ps1
```

### Method 2: Using Python Directly

```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe explore_shopcam_data.py
```

### What You'll See

The explorer will show for **each collection**:

1. **Total Documents Count**
   ```
   ✓ Total documents: 10
   ```

2. **Column Names and Types**
   ```
   📊 Columns (4): ['_id', 'nom', 'ville', 'age']
   📊 Data types:
     _id: int64
     nom: str
     ville: str
     age: int64
   ```

3. **Sample Data** (First 5 records)
   ```
   📋 Sample data:
      _id       nom    ville  age
   0    1  Client_1   Douala   21
   1    2  Client_2  Yaoundé   22
   ...
   ```

4. **Statistics** (For numeric columns)
   ```
   📈 Numeric column statistics:
               _id        age
   count  10.00000  10.000000
   mean    5.50000  25.500000
   std     3.02765   3.027650
   ...
   ```

5. **Unique Values** (For categorical columns)
   ```
   📊 Categorical columns unique values:
     ville: 2 unique values
       Values: ['Douala', 'Yaoundé']
   ```

6. **Missing Values Check**
   ```
   🔍 Missing values:
     No missing values ✓
   ```

### Understanding the Output

**For clients collection:**
- You have 10 clients
- 2 cities: Douala and Yaoundé
- Ages range from 21-30
- No missing data ✅

**For commandes collection:**
- You have 15 orders
- Order quantities: 1, 2, or 3 items
- Deliveries to 2 cities
- Links clients to products

**For produits collection:**
- You have 8 products
- Prices range: 15,000 - 350,000 CFA
- Product categories: Electronics

---

## 🎓 Training Your First Model

### Overview of Training Process

```
┌─────────────────────┐
│ 1. Connect to DB    │
│    (MongoDB Atlas)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 2. Fetch Data       │
│    (10-15 records)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 3. Preprocess       │
│    - Clean data     │
│    - Encode labels  │
│    - Scale features │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 4. Split Data       │
│    - 80% Training   │
│    - 20% Testing    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 5. Train Model      │
│    (Random Forest)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 6. Evaluate         │
│    - Accuracy       │
│    - Metrics        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 7. Save Model       │
│    (.pkl file)      │
└─────────────────────┘
```

### Method 1: Interactive Training (RECOMMENDED)

#### Step 1: Launch Training Script

**Option A: Double-click** `run_training.ps1` in Windows Explorer

**Option B: Run in PowerShell:**
```powershell
.\run_training.ps1
```

**Option C: Direct Python command:**
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe train_shopcam_sklearn.py
```

#### Step 2: Answer Prompts

**First prompt:**
```
💡 RECOMMENDATION:
First run: python explore_shopcam_data.py
This shows you all columns in your collections.

Continue with training? (y/n):
```

Type `y` and press Enter

**Second prompt - Collection Menu:**
```
============================================================
SHOPCAM ML MODEL TRAINER (Scikit-learn)
Python 3.14 Compatible - No TensorFlow required
============================================================

Available Collections:
1. clients   (10 documents)
2. commandes (15 documents)
3. produits
4. Train all collections
5. Exit

Select collection to train (1-5):
```

Choose a number (1, 2, 3, or 4) and press Enter

#### Step 3: Watch Training Progress

You'll see detailed output:

```
============================================================
TRAINING MODEL ON CLIENTS COLLECTION
============================================================

✓ Connected to MongoDB: shopcam.clients
Fetching data from clients...
✓ Fetched 10 records from MongoDB
✓ Columns: ['_id', 'nom', 'ville', 'age']

✓ Data shape: (10, 4)
✓ Columns: ['_id', 'nom', 'ville', 'age']

--- Data Preprocessing ---
✓ Target classes: ['Douala' 'Yaoundé']
✓ Training samples: 8
✓ Testing samples: 2
✓ Features: 1

--- Building RANDOM_FOREST Model ---
✓ Model created: RandomForestClassifier

--- Training Model ---
✓ Training score: 1.0000
✓ Test score: 1.0000
✓ Cross-validation score: 0.8750 (+/- 0.2165)

✓ Training completed!

--- Model Evaluation ---
Test Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

      Douala       1.00      1.00      1.00         1
     Yaoundé       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
```

#### Step 4: Review Generated Files

After training completes, check for these files:

```powershell
ls *.pkl, *.png, *.json
```

**You should see:**
- ✅ `clients_ml_model.pkl` - Your trained model
- ✅ `scaler.pkl` - Feature scaler
- ✅ `label_encoder.pkl` - Label encoder
- ✅ `model_metadata.json` - Model information
- ✅ `feature_importance.png` - Feature importance chart
- ✅ `confusion_matrix.png` - Classification results

### Method 2: Programmatic Training

Create a custom training script:

```python
from ml_mongodb_model import MongoDBMLPredictor

# Initialize (automatically reads from .env)
predictor = MongoDBMLPredictor(
    database_name='shopcam',
    collection_name='clients',
    task_type='classification'
)

# Connect
if predictor.connect_to_mongodb():
    # Fetch data
    df = predictor.fetch_data()
    
    # Preprocess
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        df,
        target_column='ville',
        drop_columns=['_id', 'nom']
    )
    
    # Build model
    predictor.build_model(model_type='random_forest')
    
    # Train
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    # Feature importance
    predictor.get_feature_importance()
    
    # Plot results
    predictor.plot_results(y_test, results['predictions'])
    
    # Save
    predictor.save_model('my_custom_model.pkl')
```

---

## 📊 Understanding the Results

### 1. Console Output Metrics

#### For Classification Tasks (e.g., predicting ville)

**Accuracy:**
```
Test Accuracy: 1.0000
```
- Range: 0.0 to 1.0 (0% to 100%)
- 1.0 = Perfect predictions
- 0.9 = 90% correct
- 0.5 = Random guessing

**Classification Report:**
```
              precision    recall  f1-score   support

      Douala       1.00      1.00      1.00         1
     Yaoundé       1.00      1.00      1.00         1
```

**Metrics Explained:**
- **Precision**: Of predictions for a class, how many were correct?
  - Example: If predicted 10 "Douala", how many were actually Douala?
  
- **Recall**: Of actual cases, how many did we find?
  - Example: Of all actual Douala clients, how many did we identify?
  
- **F1-Score**: Harmonic mean of precision and recall
  - Balance between precision and recall
  
- **Support**: Number of actual occurrences in test data

**Cross-Validation Score:**
```
✓ Cross-validation score: 0.8750 (+/- 0.2165)
```
- Tests model on multiple data splits
- More reliable than single test score
- Shows average performance and variance

#### For Regression Tasks (e.g., predicting prix)

**Mean Squared Error (MSE):**
```
Test MSE: 1234.5678
```
- Average of squared errors
- Lower is better
- Penalizes large errors heavily

**Mean Absolute Error (MAE):**
```
Test MAE: 123.45
```
- Average absolute difference
- In same units as target variable
- More interpretable than MSE

**R² Score:**
```
Test R² Score: 0.8500
```
- Range: -∞ to 1.0
- 1.0 = Perfect fit
- 0.85 = Model explains 85% of variance
- 0.0 = No better than predicting average
- Negative = Worse than predicting average

### 2. Visual Outputs

#### Feature Importance Chart (`feature_importance.png`)

**What it shows:**
- Which features the model considers most important
- Higher bars = more important for predictions

**Example:**
```
age          ████████████████████ 0.85
ville_Douala ████████ 0.15
```

**Interpretation:**
- Age is the most important feature (85%)
- Ville contributes less (15%)

#### Confusion Matrix (`confusion_matrix.png`)

**For Classification Only**

**What it shows:**
- True vs Predicted classifications
- Diagonal = Correct predictions
- Off-diagonal = Mistakes

**Example Matrix:**
```
              Predicted
              Douala  Yaoundé
Actual Douala    5       0
       Yaoundé   0       5
```

**Reading it:**
- 5 Douala correctly predicted as Douala ✅
- 0 Douala wrongly predicted as Yaoundé ✅
- 0 Yaoundé wrongly predicted as Douala ✅
- 5 Yaoundé correctly predicted as Yaoundé ✅
- Perfect classification!

#### Predictions vs Actual (`predictions_vs_actual.png`)

**For Regression Only**

**What it shows:**
- Scatter plot of predictions vs actual values
- Red diagonal line = perfect predictions
- Points close to line = good predictions

**Interpretation:**
- Points on red line = perfect prediction
- Points above line = over-prediction
- Points below line = under-prediction
- Tighter clustering = better model

### 3. Saved Model Files

#### `clients_ml_model.pkl`

**Contains:**
- Trained Random Forest model
- All learned patterns
- Ready for making predictions

**Size:** ~10-100 KB (small dataset)

**Can be loaded later:**
```python
import joblib
model = joblib.load('clients_ml_model.pkl')
```

#### `scaler.pkl`

**Contains:**
- Feature scaling parameters
- Mean and standard deviation for each feature

**Why needed:**
- New data must be scaled the same way
- Ensures consistent predictions

#### `label_encoder.pkl`

**Contains:**
- Mapping of text labels to numbers
- Example: {'Douala': 0, 'Yaoundé': 1}

**Why needed:**
- Convert predictions back to text
- Example: 0 → 'Douala'

#### `model_metadata.json`

**Contains:**
```json
{
    "task_type": "classification",
    "feature_names": ["age"],
    "model_type": "RandomForestClassifier",
    "timestamp": "2026-02-02T15:30:00",
    "training_history": {
        "train_score": 1.0,
        "test_score": 1.0,
        "cv_scores": [0.75, 1.0, 0.75, 1.0, 1.0]
    }
}
```

**Useful for:**
- Tracking model versions
- Understanding model configuration
- Debugging issues

---

## 🔮 Making Predictions

### Using Trained Model

#### Step 1: Load the Model

Create a new Python file `make_predictions.py`:

```python
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Load saved artifacts
model = joblib.load('clients_ml_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

print("✓ Model loaded successfully!")
```

#### Step 2: Prepare New Data

```python
# Single prediction
new_client = {
    'age': 28
}

# Convert to DataFrame
new_df = pd.DataFrame([new_client])

print("Input data:")
print(new_df)
```

#### Step 3: Scale Features

```python
# Scale the data (same way as training)
new_scaled = scaler.transform(new_df)

print("Scaled data:")
print(new_scaled)
```

#### Step 4: Make Prediction

```python
# Predict
prediction_encoded = model.predict(new_scaled)

# Decode to original labels
prediction = label_encoder.inverse_transform(prediction_encoded)

print(f"\n🎯 Prediction: {prediction[0]}")
```

#### Step 5: Get Prediction Probabilities

```python
# Get probabilities for each class
probabilities = model.predict_proba(new_scaled)

print("\n📊 Prediction Probabilities:")
for label, prob in zip(label_encoder.classes_, probabilities[0]):
    print(f"  {label}: {prob:.2%}")
```

### Complete Prediction Script

Save as `predict_new.py`:

```python
"""
Make predictions on new data using trained model
"""
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def predict_client_city(age):
    """Predict client city based on age"""
    
    # Load model artifacts
    model = joblib.load('clients_ml_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Prepare data
    data = pd.DataFrame([{'age': age}])
    
    # Scale
    data_scaled = scaler.transform(data)
    
    # Predict
    prediction_encoded = model.predict(data_scaled)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(data_scaled)[0]
    
    # Display results
    print(f"\n🎯 Prediction for age {age}:")
    print(f"   City: {prediction}")
    print(f"\n📊 Confidence:")
    for label, prob in zip(label_encoder.classes_, probabilities):
        bar = "█" * int(prob * 20)
        print(f"   {label:10} {bar} {prob:.1%}")
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Test with different ages
    ages_to_test = [22, 25, 28, 30]
    
    for age in ages_to_test:
        predict_client_city(age)
        print("-" * 50)
```

Run it:
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe predict_new.py
```

### Batch Predictions

For multiple predictions at once:

```python
import pandas as pd
import joblib

# Load model
model = joblib.load('clients_ml_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Multiple new clients
new_clients = pd.DataFrame({
    'age': [22, 24, 26, 28, 30]
})

# Scale
new_clients_scaled = scaler.transform(new_clients)

# Predict
predictions_encoded = model.predict(new_clients_scaled)
predictions = label_encoder.inverse_transform(predictions_encoded)

# Add to dataframe
new_clients['predicted_ville'] = predictions

print(new_clients)
```

**Output:**
```
   age predicted_ville
0   22         Yaoundé
1   24         Yaoundé
2   26          Douala
3   28          Douala
4   30          Douala
```

---

## 🔧 Troubleshooting

### Problem 1: Python Not Found Error

**Error Message:**
```
[ERROR] Failed to launch 'C:\Users\landm\AppData\Local\Programs\Python\Python312\python.exe' (0x80070003)
```

**Cause:** System trying to use wrong Python version

**Solution:**
Always use full Python path:
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe script_name.py
```

Or use PowerShell scripts:
```powershell
.\run_training.ps1
.\run_explore.ps1
```

---

### Problem 2: MongoDB Connection Failed

**Error Message:**
```
✗ Error connecting to MongoDB: ...
```

**Possible Causes & Solutions:**

1. **Check .env file exists:**
   ```powershell
   cat .env
   ```
   Should show `MONGO_DB_STRING=...`

2. **Verify MongoDB URI:**
   ```powershell
   C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('MONGO_DB_STRING'))"
   ```

3. **Test connection manually:**
   ```python
   from pymongo import MongoClient
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   uri = os.getenv('MONGO_DB_STRING')
   
   try:
       client = MongoClient(uri)
       client.server_info()
       print("✓ Connection successful!")
   except Exception as e:
       print(f"✗ Connection failed: {e}")
   ```

4. **Check internet connection:**
   ```powershell
   Test-Connection -ComputerName google.com -Count 2
   ```

5. **Verify MongoDB Atlas status:**
   - Go to https://cloud.mongodb.com/
   - Check if cluster is running

---

### Problem 3: Module Not Found Error

**Error Message:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
Install missing package:
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip install scikit-learn
```

For all packages:
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip install scikit-learn pandas numpy pymongo matplotlib seaborn joblib scipy python-dotenv
```

---

### Problem 4: Target Column Not Found

**Error Message:**
```
✗ Error: Target column 'status' not found!
Available columns: ['_id', 'nom', 'ville', 'age']
```

**Cause:** Script using wrong column name

**Solution:**
1. Run data exploration:
   ```powershell
   .\run_explore.ps1
   ```

2. Note the actual column names

3. Edit `train_shopcam_sklearn.py`:
   - Find the function for your collection
   - Update `TARGET_COLUMN` with correct name
   - Example: Change `'status'` to `'ville'`

---

### Problem 5: Insufficient Data Warning

**Warning Message:**
```
UserWarning: The least populated class in y has only 1 members
```

**Cause:** Very small dataset (8-15 samples)

**Impact:**
- Model may overfit
- Test accuracy might be misleading
- Not reliable for production

**Solutions:**

1. **Reduce test split:**
   ```python
   # In ml_mongodb_model.py, change:
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.1, random_state=42  # Changed from 0.2 to 0.1
   )
   ```

2. **Use leave-one-out cross-validation:**
   ```python
   from sklearn.model_selection import LeaveOneOut
   ```

3. **Collect more data** (recommended)

4. **Use simpler models:**
   - Change `n_estimators` to 10-50
   - Reduce `max_depth` to 3-5

---

### Problem 6: Perfect Accuracy (Too Good to Be True)

**Output:**
```
Test Accuracy: 1.0000
```

**Possible Causes:**

1. **Data Leakage:**
   - Including the target in features
   - Using future information
   
   **Check:** Verify `drop_columns` includes target-related columns

2. **Overfitting:**
   - Model memorized training data
   - Dataset too small
   
   **Solution:** Check cross-validation score

3. **Simple Problem:**
   - Data has obvious patterns
   - Actually achievable with small dataset

**How to Verify:**
```python
# Check cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV scores: {cv_scores}")
print(f"Mean: {cv_scores.mean()}, Std: {cv_scores.std()}")
```

If CV scores are also perfect (all 1.0), problem might be too simple for this data.

---

### Problem 7: Import Errors After Installation

**Error:**
```
ImportError: DLL load failed while importing _multiarray_umath
```

**Cause:** Conflicting numpy versions or corrupted installation

**Solution:**
```powershell
# Uninstall numpy
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip uninstall numpy -y

# Reinstall
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe -m pip install numpy --no-cache-dir
```

---

### Problem 8: PowerShell Execution Policy

**Error:**
```
.\run_training.ps1 : File cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
# Check current policy
Get-ExecutionPolicy

# Set to allow scripts (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run directly
powershell -ExecutionPolicy Bypass -File .\run_training.ps1
```

---

## 🚀 Advanced Usage

### 1. Custom Model Configuration

Edit training parameters in the script:

```python
def train_clients_model():
    # ... existing code ...
    
    # Custom Random Forest parameters
    from sklearn.ensemble import RandomForestClassifier
    
    custom_model = RandomForestClassifier(
        n_estimators=200,        # More trees
        max_depth=15,            # Deeper trees
        min_samples_split=2,     # Minimum samples to split
        min_samples_leaf=1,      # Minimum samples per leaf
        max_features='sqrt',     # Features per split
        random_state=42,
        n_jobs=-1,              # Use all CPU cores
        verbose=1               # Show progress
    )
    
    predictor.model = custom_model
```

### 2. Feature Engineering

Add custom features before training:

```python
# In train_shopcam_sklearn.py, after fetching data:

# Add age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 30, 100], 
                         labels=['Young', 'Adult', 'Senior'])

# Add age squared (polynomial feature)
df['age_squared'] = df['age'] ** 2

# Interaction features
df['age_times_id'] = df['age'] * df['_id']
```

### 3. Hyperparameter Tuning

Find best parameters automatically:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
```

### 4. Ensemble Methods

Combine multiple models:

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Define models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

# Voting classifier
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft'  # Use probabilities
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
score = ensemble.score(X_test, y_test)
print(f"Ensemble accuracy: {score}")
```

### 5. Save Training History

Track all training runs:

```python
import json
from datetime import datetime

# After training
history = {
    'timestamp': datetime.now().isoformat(),
    'collection': 'clients',
    'target': 'ville',
    'model_type': 'RandomForest',
    'train_accuracy': train_score,
    'test_accuracy': test_score,
    'cv_scores': cv_scores.tolist(),
    'parameters': model.get_params()
}

# Append to history file
try:
    with open('training_history.json', 'r') as f:
        all_history = json.load(f)
except FileNotFoundError:
    all_history = []

all_history.append(history)

with open('training_history.json', 'w') as f:
    json.dump(all_history, f, indent=4)

print("Training history saved!")
```

### 6. Cross-Database Training

Train on data from multiple collections:

```python
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

client = MongoClient(os.getenv('MONGO_DB_STRING'))
db = client['shopcam']

# Fetch from multiple collections
clients = pd.DataFrame(list(db.clients.find()))
commandes = pd.DataFrame(list(db.commandes.find()))

# Merge on client_id
merged = commandes.merge(clients, left_on='client_id', right_on='_id', 
                        suffixes=('_order', '_client'))

# Now have richer features
print(merged.columns)
# Can predict based on both order and client data!
```

### 7. Model Comparison

Compare different algorithms:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Score
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    results[name] = {
        'train_score': train_score,
        'test_score': test_score
    }
    
    print(f"{name}:")
    print(f"  Train: {train_score:.4f}")
    print(f"  Test:  {test_score:.4f}")
    print()

# Find best model
best_model = max(results, key=lambda x: results[x]['test_score'])
print(f"Best model: {best_model}")
```

---

## 📚 Best Practices

### 1. Data Management

✅ **DO:**
- Regularly backup your database
- Version your data
- Document data changes
- Keep raw and processed data separate
- Use consistent naming conventions

❌ **DON'T:**
- Modify source data directly
- Mix test and training data
- Ignore missing values
- Use hardcoded paths

### 2. Model Development

✅ **DO:**
- Always split train/test data
- Use cross-validation
- Save model metadata
- Version your models
- Document model parameters
- Track performance over time

❌ **DON'T:**
- Train on test data
- Ignore overfitting
- Skip evaluation
- Forget to save models
- Use default parameters blindly

### 3. Code Organization

✅ **DO:**
- Use version control (Git)
- Write modular code
- Add comments
- Use configuration files
- Handle errors gracefully

❌ **DON'T:**
- Hardcode credentials
- Write monolithic scripts
- Ignore errors
- Skip documentation

### 4. Security

✅ **DO:**
- Use .env files for credentials
- Add .env to .gitignore
- Use environment variables
- Restrict database access
- Regularly rotate passwords

❌ **DON'T:**
- Commit credentials to Git
- Share credentials publicly
- Use plain text passwords
- Grant unnecessary permissions

### 5. Production Deployment

✅ **DO:**
- Test thoroughly
- Monitor performance
- Log predictions
- Handle edge cases
- Set up alerts

❌ **DON'T:**
- Deploy untested models
- Ignore errors in production
- Skip monitoring
- Forget about maintenance

---

## ❓ FAQ

### Q1: Why is my accuracy 100%?

**A:** With such small datasets (8-15 samples), perfect accuracy is common but may indicate:
- Simple, perfectly separable patterns
- Overfitting (model memorized data)
- Data leakage (target information in features)

**Check:** Cross-validation scores and feature importance

### Q2: How much data do I need?

**A:** General guidelines:
- **Minimum**: 100 samples per class
- **Good**: 1,000+ samples
- **Excellent**: 10,000+ samples

**Your situation:** 8-15 samples is very small. Good for learning, not production.

### Q3: Should I use CNN or Random Forest?

**A:** For your data size:
- ✅ **Random Forest**: Works great with small data
- ❌ **CNN**: Requires thousands of samples, better for images/sequences

**Current solution:** Random Forest is the right choice!

### Q4: How do I collect more data?

**Options:**
1. Generate synthetic data (use with caution)
2. Collect real data over time
3. Use data augmentation techniques
4. Combine multiple data sources

### Q5: Can I use this for production?

**Current state:** No, because:
- Too few samples (8-15)
- No validation on unseen data
- Limited features

**To make production-ready:**
1. Collect 100+ samples minimum
2. Add more features
3. Implement monitoring
4. Add error handling
5. Set up logging
6. Create API endpoint

### Q6: What if MongoDB is down?

**Solution:** Add error handling:
```python
try:
    if predictor.connect_to_mongodb():
        # Training code
    else:
        print("Connection failed, using cached data...")
        # Load from local backup
except Exception as e:
    print(f"Error: {e}")
    # Send alert
    # Use fallback
```

### Q7: How do I update the model with new data?

**Option 1: Retrain from scratch** (recommended)
```python
# Fetch all data including new
df = predictor.fetch_data()

# Retrain completely
predictor.train(X_train, y_train, X_test, y_test)
predictor.save_model('clients_ml_model_v2.pkl')
```

**Option 2: Incremental learning**
```python
# Some models support partial_fit
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
model.partial_fit(new_X, new_y, classes=np.unique(y))
```

### Q8: Can I train on my own computer without internet?

**A:** Not with MongoDB Atlas. Options:
1. **Install local MongoDB:**
   ```powershell
   # Download from mongodb.com
   # Import data locally
   mongoimport --db shopcam --collection clients --file clients.json
   ```

2. **Export data to CSV:**
   ```python
   df.to_csv('clients_backup.csv', index=False)
   # Then train from CSV
   df = pd.read_csv('clients_backup.csv')
   ```

### Q9: How do I deploy as a web service?

**Basic Flask API:**
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model once at startup
model = joblib.load('clients_ml_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    age = data['age']
    
    # Prepare
    import pandas as pd
    df = pd.DataFrame([{'age': age}])
    df_scaled = scaler.transform(df)
    
    # Predict
    pred = model.predict(df_scaled)
    result = label_encoder.inverse_transform(pred)[0]
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
```

### Q10: What's next after this tutorial?

**Learning path:**
1. ✅ Complete this tutorial
2. Experiment with different models
3. Learn feature engineering
4. Study cross-validation techniques
5. Explore deep learning (when you have more data)
6. Build a web application
7. Deploy to cloud (AWS, Azure, GCP)

**Resources:**
- Scikit-learn documentation
- Kaggle competitions
- Machine Learning courses (Coursera, edX)
- Real-world projects

---

## 🎓 Summary

### What You've Learned

1. ✅ **Setup & Installation**
   - Python environment
   - Required packages
   - MongoDB connection

2. ✅ **Data Exploration**
   - Viewing database structure
   - Understanding data types
   - Identifying patterns

3. ✅ **Model Training**
   - Data preprocessing
   - Model selection
   - Training process
   - Evaluation metrics

4. ✅ **Making Predictions**
   - Loading saved models
   - Preparing new data
   - Getting predictions

5. ✅ **Troubleshooting**
   - Common errors
   - Solutions
   - Best practices

### Your Complete Toolkit

**Scripts:**
- ✅ `run_training.ps1` - Easy training
- ✅ `run_explore.ps1` - Data exploration
- ✅ `train_shopcam_sklearn.py` - Custom training
- ✅ `ml_mongodb_model.py` - ML model class

**Models:**
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ XGBoost (optional)

**Outputs:**
- ✅ Trained models (.pkl)
- ✅ Visualizations (.png)
- ✅ Metrics (JSON)

### Next Steps

1. **Practice:**
   - Train on all 3 collections
   - Experiment with parameters
   - Try different models

2. **Expand:**
   - Collect more data
   - Add more features
   - Improve accuracy

3. **Deploy:**
   - Create API
   - Build web interface
   - Monitor performance

---

## 📞 Support & Resources

### Documentation
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/docs/)
- [PyMongo](https://pymongo.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)

### Community
- Stack Overflow
- Reddit r/MachineLearning
- GitHub Discussions

### Your Files
- `START_HERE.md` - Quick start
- `DATA_STRUCTURE.md` - Database info
- `SHOPCAM_QUICKSTART.md` - Reference guide
- This file - Complete documentation

---

## ✅ Checklist

Before you start:
- [ ] Python 3.14 installed and working
- [ ] All packages installed (`pip list`)
- [ ] `.env` file in project folder
- [ ] MongoDB connection tested
- [ ] Data explored (`run_explore.ps1`)

Ready to train:
- [ ] Choose collection (clients, commandes, or produits)
- [ ] Identify target column
- [ ] Understand what you're predicting
- [ ] Run `run_training.ps1`

After training:
- [ ] Check accuracy metrics
- [ ] Review visualizations
- [ ] Understand feature importance
- [ ] Save model for predictions

---

**🎉 You're now ready to train machine learning models on your MongoDB data!**

**Start here:** Run `.\run_training.ps1` and select option 1 (clients)

**Questions?** Review the FAQ section above.

**Good luck! 🚀**

---

*Last updated: February 2, 2026*
*Version: 1.0*
*Compatible with: Python 3.8 - 3.14*
