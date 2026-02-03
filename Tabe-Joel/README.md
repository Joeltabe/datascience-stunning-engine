# CNN Model for MongoDB Data - Predictive Insights System

A comprehensive deep learning system that connects to MongoDB, trains a Convolutional Neural Network (CNN), and provides predictive insights for both classification and regression tasks.

## 🚀 Features

- **MongoDB Integration**: Direct connection to MongoDB databases for seamless data retrieval
- **Flexible CNN Architecture**: Automatically adapts to classification or regression tasks
- **Advanced Preprocessing**: Comprehensive data cleaning, feature engineering, and transformation
- **Predictive Insights**: Detailed model evaluation with visualizations and metrics
- **Production-Ready**: Save/load models, batch predictions, and command-line interface
- **Visualization**: Training history plots, confusion matrices, and prediction analysis

## 📋 Prerequisites

- Python 3.8 or higher
- MongoDB instance (local or cloud)
- 4GB+ RAM recommended
- GPU optional (for faster training)

## 🔧 Installation

### 1. Clone or Download Files

Ensure you have all the following files in your directory:
- `cnn_mongodb_model.py` - Main training script
- `config.py` - Configuration settings
- `data_preprocessing.py` - Data preprocessing module
- `predict.py` - Prediction script
- `requirements.txt` - Dependencies

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install tensorflow keras numpy pandas scikit-learn pymongo matplotlib seaborn joblib imbalanced-learn scipy
```

### 3. MongoDB Setup

**Option A: Local MongoDB**
```bash
# Install MongoDB Community Edition
# Start MongoDB service
mongod --dbpath /path/to/data
```

**Option B: MongoDB Atlas (Cloud)**
1. Create account at https://www.mongodb.com/cloud/atlas
2. Create cluster and database
3. Get connection string
4. Update `config.py` with your URI

## ⚙️ Configuration

Edit `config.py` to match your setup:

```python
MONGODB_CONFIG = {
    'uri': 'mongodb://localhost:27017/',  # Your MongoDB URI
    'database': 'your_database_name',
    'collection': 'your_collection_name',
    'query': {},  # Optional filter
    'limit': None  # None = fetch all
}

MODEL_CONFIG = {
    'task_type': 'classification',  # or 'regression'
    'target_column': 'target',  # Your target variable
    'drop_columns': ['id'],  # Columns to exclude
    'epochs': 100,
    'batch_size': 32
}
```

## 🎯 Quick Start

### Example 1: Classification Task

```python
from cnn_mongodb_model import MongoDBCNNPredictor

# Initialize
predictor = MongoDBCNNPredictor(
    mongo_uri="mongodb://localhost:27017/",
    database_name="customer_db",
    collection_name="customer_data",
    task_type="classification"
)

# Connect and fetch data
predictor.connect_to_mongodb()
df = predictor.fetch_data(limit=10000)

# Preprocess
X_train, X_test, y_train, y_test = predictor.preprocess_data(
    df, 
    target_column='churn',
    drop_columns=['customer_id', 'signup_date']
)

# Build and train
predictor.build_cnn_model(
    input_shape=(X_train.shape[1], 1),
    num_classes=2
)
predictor.train(X_train, y_train, X_test, y_test)

# Evaluate and get insights
results = predictor.evaluate(X_test, y_test)
insights = predictor.get_insights(X_test, y_test)

# Save model
predictor.save_model()
```

### Example 2: Regression Task

```python
predictor = MongoDBCNNPredictor(
    mongo_uri="mongodb://localhost:27017/",
    database_name="sales_db",
    collection_name="sales_data",
    task_type="regression"
)

# Same workflow as above...
```

## 📊 Using the Preprocessor

For advanced data preprocessing:

```python
from data_preprocessing import MongoDBDataPreprocessor

preprocessor = MongoDBDataPreprocessor(
    mongo_uri="mongodb://localhost:27017/",
    database_name="mydb",
    collection_name="mycollection"
)

preprocessor.connect()

# Full pipeline
X, y, feature_names = preprocessor.full_preprocessing_pipeline(
    query={'year': {'$gte': 2020}},
    target_column='sales',
    drop_columns=['id', 'timestamp'],
    datetime_column='date',
    handle_outliers=True,
    balance_data=False
)

preprocessor.disconnect()
```

## 🔮 Making Predictions

### Command-Line Interface

**Batch predictions:**
```bash
python predict.py --input new_data.csv --output predictions.csv
```

**Interactive mode:**
```bash
python predict.py --single
```

**Feature importance:**
```bash
python predict.py --importance
```

### Programmatic Usage

```python
from predict import CNNPredictor

# Load trained model
predictor = CNNPredictor(model_path='cnn_model.h5')
predictor.load_artifacts()

# Single prediction
sample = {
    'age': 35,
    'income': 50000,
    'credit_score': 720
}
prediction = predictor.predict_single(sample)

# Batch predictions
import pandas as pd
data = pd.read_csv('new_customers.csv')
predictions_df = predictor.predict_batch(data)

# Save predictions
predictor.save_predictions(data, 'customer_predictions.csv')
```

## 📁 Project Structure

```
Tabe Joel/
├── cnn_mongodb_model.py      # Main CNN training script
├── config.py                  # Configuration settings
├── data_preprocessing.py      # Data preprocessing utilities
├── predict.py                 # Prediction script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── Generated Files (after training):
├── cnn_model.h5              # Trained model
├── best_cnn_model.h5         # Best model checkpoint
├── scaler.pkl                # Feature scaler
├── label_encoder.pkl         # Label encoder (classification)
├── model_metadata.json       # Model metadata
├── training_history.png      # Training plots
├── confusion_matrix.png      # Confusion matrix (classification)
└── predictions_vs_actual.png # Prediction plot (regression)
```

## 🎨 Outputs and Visualizations

### Training Outputs
- **training_history.png**: Loss and accuracy/MAE curves
- **Model checkpoints**: Best model saved during training

### Classification Outputs
- **confusion_matrix.png**: Visual confusion matrix
- **Classification report**: Precision, recall, F1-score
- **Class probabilities**: Confidence scores

### Regression Outputs
- **predictions_vs_actual.png**: Scatter plot of predictions
- **R² score**: Model fit quality
- **MAE/MSE**: Error metrics
- **MAPE**: Percentage error

## 🔍 Example Use Cases

### 1. Customer Churn Prediction
```python
# config.py
MONGODB_CONFIG = {
    'database': 'telecom_db',
    'collection': 'customers'
}
MODEL_CONFIG = {
    'task_type': 'classification',
    'target_column': 'churned'
}
```

### 2. Sales Forecasting
```python
MODEL_CONFIG = {
    'task_type': 'regression',
    'target_column': 'monthly_sales'
}
```

### 3. Stock Price Prediction
```python
MONGODB_CONFIG = {
    'collection': 'stock_prices',
    'query': {'symbol': 'AAPL'}
}
MODEL_CONFIG = {
    'task_type': 'regression',
    'target_column': 'close_price'
}
```

### 4. Disease Diagnosis
```python
MODEL_CONFIG = {
    'task_type': 'classification',
    'target_column': 'diagnosis'
}
```

## ⚡ Performance Tips

1. **Data Size**: Start with 10,000-50,000 samples for testing
2. **Feature Engineering**: Use `data_preprocessing.py` for better results
3. **Batch Size**: Adjust based on available memory (16, 32, 64, 128)
4. **Early Stopping**: Model stops when validation loss stops improving
5. **GPU Acceleration**: TensorFlow automatically uses GPU if available

## 🐛 Troubleshooting

### MongoDB Connection Issues
```python
# Check MongoDB is running
mongod --version

# Test connection
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
print(client.list_database_names())
```

### Memory Issues
```python
# Reduce batch size
MODEL_CONFIG['batch_size'] = 16

# Limit data
MONGODB_CONFIG['limit'] = 10000

# Use data generator (for very large datasets)
```

### Model Not Converging
```python
# Try different learning rate
MODEL_CONFIG['learning_rate'] = 0.0001

# More epochs
MODEL_CONFIG['epochs'] = 200

# Check data quality
preprocessor.get_data_info(df)
```

## 📈 Model Evaluation Metrics

### Classification
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: Actual positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class predictions

### Regression
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Variance explained by model
- **MAPE**: Mean Absolute Percentage Error

## 🔐 Security Best Practices

1. **Environment Variables**: Store credentials in `.env` file
```python
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv('MONGODB_URI')
```

2. **Don't commit credentials**: Add to `.gitignore`:
```
.env
*.pkl
*.h5
model_metadata.json
```

3. **Use MongoDB Atlas**: Encrypted connections by default

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [MongoDB Python Driver](https://pymongo.readthedocs.io/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Keras CNN Guide](https://keras.io/guides/)

## 🤝 Contributing

Feel free to extend this system:
- Add more CNN architectures (LSTM, Transformer)
- Implement hyperparameter tuning
- Add real-time prediction API
- Integrate with other databases

## 📝 License

This project is for educational purposes. Modify and use as needed.

## ✨ Features Roadmap

- [ ] Hyperparameter optimization (Grid Search, Bayesian)
- [ ] REST API for predictions
- [ ] Real-time streaming predictions
- [ ] Multi-output models
- [ ] Transfer learning support
- [ ] AutoML integration
- [ ] Docker containerization
- [ ] Model versioning

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review MongoDB and TensorFlow documentation
3. Verify data format and configuration

---

**Happy Predicting! 🎯**
