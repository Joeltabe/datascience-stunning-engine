# Shopcam CNN Model - Quick Start Guide

## 🎯 Your Database Configuration

**MongoDB Atlas Connection:**
- URI: `mongodb+srv://lmuij113_db_user:***@cluster0.y3swaud.mongodb.net/`
- Database: `shopcam`

**Collections:**
1. **clients** - 10 documents (60 bytes avg)
2. **commandes** - 15 documents (87 bytes avg)  
3. **produits** - ? documents

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Explore Your Data
```bash
python explore_shopcam_data.py
```

This will show you:
- All columns in each collection
- Sample data
- Data types
- Statistics

### Step 3: Train Your Model
```bash
python train_shopcam_model.py
```

Choose which collection to train on and follow the prompts!

---

## 📊 Understanding Your Collections

### Clients Collection
- **Use Case**: Client segmentation, status prediction
- **Possible Targets**: client type, status, category
- **Model Type**: Usually classification

### Commandes Collection  
- **Use Case**: Order amount prediction, status forecasting
- **Possible Targets**: montant (amount), status
- **Model Type**: Regression (for amounts) or Classification (for status)

### Produits Collection
- **Use Case**: Price prediction, category classification
- **Possible Targets**: prix (price), category
- **Model Type**: Regression (for price) or Classification (for category)

---

## ⚙️ Configuration Files

All configuration is in [config.py](config.py):

```python
MONGODB_CONFIG = {
    'uri': 'mongodb+srv://lmuij113_db_user:***@cluster0.y3swaud.mongodb.net/',
    'database': 'shopcam',
    'collection': 'clients',  # Change as needed
}
```

---

## 📝 Example Workflows

### Example 1: Predict Client Status
```python
from cnn_mongodb_model import MongoDBCNNPredictor

predictor = MongoDBCNNPredictor(
    mongo_uri="mongodb+srv://lmuij113_db_user:***@cluster0.y3swaud.mongodb.net/",
    database_name="shopcam",
    collection_name="clients",
    task_type="classification"
)

# Connect and train
predictor.connect_to_mongodb()
df = predictor.fetch_data()
X_train, X_test, y_train, y_test = predictor.preprocess_data(
    df, target_column='status', drop_columns=['_id']
)
predictor.build_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=2)
predictor.train(X_train, y_train, X_test, y_test)
predictor.save_model('clients_model.h5')
```

### Example 2: Predict Order Amount
```python
predictor = MongoDBCNNPredictor(
    mongo_uri="mongodb+srv://lmuij113_db_user:***@cluster0.y3swaud.mongodb.net/",
    database_name="shopcam",
    collection_name="commandes",
    task_type="regression"
)

# Train on commandes data...
predictor.connect_to_mongodb()
df = predictor.fetch_data()
X_train, X_test, y_train, y_test = predictor.preprocess_data(
    df, target_column='montant', drop_columns=['_id']
)
predictor.build_cnn_model(input_shape=(X_train.shape[1], 1))
predictor.train(X_train, y_train, X_test, y_test)
```

---

## 🔮 Making Predictions

After training, use your model:

```bash
# Load and predict
python predict.py --model clients_cnn_model.h5 --single
```

Or in Python:
```python
from predict import CNNPredictor

predictor = CNNPredictor('clients_cnn_model.h5')
predictor.load_artifacts()

# Predict single client
new_client = {
    'age': 35,
    'total_orders': 15,
    # ... other features
}
prediction = predictor.predict_single(new_client)
```

---

## 📈 Expected Outputs

After training, you'll get:
- ✅ `clients_cnn_model.h5` - Trained model
- ✅ `clients_scaler.pkl` - Feature scaler
- ✅ `training_history.png` - Training plots
- ✅ `confusion_matrix.png` - Classification results (if classification)
- ✅ `predictions_vs_actual.png` - Regression results (if regression)

---

## ⚠️ Important Notes

1. **Small Dataset Warning**: You have 10-15 documents per collection
   - This is very small for deep learning
   - Model may overfit
   - Consider collecting more data or using simpler models
   - Use small batch sizes (4-8) and fewer epochs (20-50)

2. **Update Target Columns**: 
   - Edit `train_shopcam_model.py`
   - Set correct `TARGET_COLUMN` for each collection
   - Based on what you see in `explore_shopcam_data.py`

3. **Data Privacy**:
   - Your MongoDB credentials are in the code
   - Don't share these files publicly
   - Consider using environment variables

---

## 🛠️ Troubleshooting

### Connection Error
```bash
# Test connection
python -c "from pymongo import MongoClient; print(MongoClient('mongodb+srv://lmuij113_db_user:***@cluster0.y3swaud.mongodb.net/').list_database_names())"
```

### Not Enough Data
- Reduce `batch_size` to 4 or 8
- Reduce `epochs` to 20-30
- Use data augmentation
- Consider collecting more data

### Column Not Found
- Run `explore_shopcam_data.py` first
- Check actual column names
- Update `TARGET_COLUMN` in training script

---

## 📞 Next Steps

1. ✅ Install dependencies
2. ✅ Run data exploration
3. ✅ Update target columns in training script
4. ✅ Train your first model
5. ✅ Make predictions
6. ✅ Collect more data if needed

---

**Ready to start? Run:**
```bash
python explore_shopcam_data.py
```
