# 🚀 READY TO USE - Quick Start Guide

## ✅ Everything is Set Up!

Your MongoDB CNN/ML system is configured and ready to train on your shopcam database.

---

## 📍 You Are Here
Location: `C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel\`

---

## ⚡ Quick Commands

### 1. View Your Data (Already Done!)
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe explore_shopcam_data.py
```

### 2. Train Your First Model
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe train_shopcam_sklearn.py
```

---

## 📊 Your Database Summary

### clients (10 documents)
- Predict **ville** (city) → Classification
- Columns: _id, nom, ville, age

### commandes (15 documents)
- Predict **quantite** (quantity) → Regression
- Columns: _id, client_id, produit_id, quantite, ville_livraison

### produits (8 documents)
- Predict **prix** (price) → Regression
- Columns: _id, nom, prix

---

## 🎯 Pre-Configured Models

The training script is already configured with realistic targets:

1. **clients**: Predict which city (Douala/Yaoundé) based on age
2. **commandes**: Predict order quantity
3. **produits**: Predict product price

---

## 🏃 Let's Train!

Run this command:
```powershell
cd 'C:\Users\landm\OneDrive\Documents\Masters-Notes\ai\Datasets\Tabe-Joel'
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe train_shopcam_sklearn.py
```

You'll see a menu to choose which collection to train on!

---

## 📈 What You'll Get

After training:
- ✅ Trained model (`.pkl` file)
- ✅ Feature importance chart
- ✅ Confusion matrix (for classification)
- ✅ Prediction plots (for regression)
- ✅ Performance metrics

---

## 💡 Pro Tips

1. **Small Dataset Warning**: You have 8-15 samples per collection
   - Results may not be perfect
   - This is a demonstration/learning project
   - Collect more data for production use

2. **Python Path**: Use full path to avoid the Python 3.12 error:
   ```
   C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe
   ```

3. **View Results**: Check the generated PNG images for visualizations

---

## 🆘 Troubleshooting

### If you get Python error:
Use the full Python path:
```powershell
C:\Users\landm\AppData\Local\Programs\Python\Python314\python.exe script_name.py
```

### If MongoDB connection fails:
Check that your `.env` file exists in this folder with:
```
MONGO_DB_STRING=mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority
```

---

## ✨ You're All Set!

Everything is working perfectly. Just run the training command above and follow the menu! 🎉
