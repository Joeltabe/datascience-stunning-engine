# Shopcam Database Structure Summary

## ✅ Successfully Connected to MongoDB Atlas!

Your database has been explored. Here's what we found:

---

## 📊 Collection 1: **clients** (10 documents)

**Columns:**
- `_id` - Client ID (1-10)
- `nom` - Client name (Client_1, Client_2, etc.)
- `ville` - City (Douala, Yaoundé)
- `age` - Age (21-30)

**Possible Predictions:**
- ✅ **Classify ville** (Classification - Predict city: Douala or Yaoundé)
- ✅ **Predict age** (Regression - Predict client age)

**Suggested Model:**
```python
TARGET = 'ville'  # Classification
TASK = 'classification'
```

---

## 📊 Collection 2: **commandes** (15 documents)

**Columns:**
- `_id` - Order ID (501-515)
- `client_id` - Client who made the order
- `produit_id` - Product ordered
- `quantite` - Quantity (1-3)
- `ville_livraison` - Delivery city (Douala, Yaoundé)

**Possible Predictions:**
- ✅ **Predict quantite** (Regression - Predict order quantity)
- ✅ **Classify ville_livraison** (Classification - Predict delivery city)
- ✅ **Predict client_id** (Classification - Identify which client)

**Suggested Model:**
```python
TARGET = 'quantite'  # Regression
TASK = 'regression'
```

---

## 📊 Collection 3: **produits** (8 documents)

**Columns:**
- `_id` - Product ID (101-108)
- `nom` - Product name (Laptop HP, Smartphone, Tablette, etc.)
- `prix` - Price (15,000 - 350,000 CFA)

**Possible Predictions:**
- ✅ **Predict prix** (Regression - Predict product price)
- ✅ **Classify nom** (Classification - Predict product category)

**Suggested Model:**
```python
TARGET = 'prix'  # Regression
TASK = 'regression'
```

---

## 🎯 Recommended Use Cases

### 1. Client City Prediction (clients)
Predict which city a client is from based on their age
```python
COLLECTION = 'clients'
TARGET = 'ville'
TASK = 'classification'
```

### 2. Order Quantity Prediction (commandes)
Predict how many items will be ordered
```python
COLLECTION = 'commandes'
TARGET = 'quantite'
TASK = 'regression'
```

### 3. Product Price Prediction (produits)
Predict product price based on product ID
```python
COLLECTION = 'produits'
TARGET = 'prix'
TASK = 'regression'
```

---

## ⚠️ Important Notes

1. **Small Dataset**: You have very few samples (8-15 documents)
   - This is too small for deep learning
   - Scikit-learn models will work better
   - Results may not be highly accurate

2. **Limited Features**: Each collection has only 3-5 columns
   - Consider joining collections for more features
   - Or collect more data

3. **Ready to Train**: Everything is set up!
   ```bash
   python train_shopcam_sklearn.py
   ```

---

## 📈 Next Steps

1. Choose a collection and target from above
2. Run the training script
3. Review the results and visualizations
4. Optionally: collect more data for better accuracy
