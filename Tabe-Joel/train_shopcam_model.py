"""
Train CNN Model on Shopcam Data
Ready-to-use training script for your shopcam database
"""

from cnn_mongodb_model import MongoDBCNNPredictor
from config import MONGODB_CONFIG
import sys

# Shopcam MongoDB Configuration
MONGO_URI = "mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority"
DATABASE = "shopcam"

def train_clients_model():
    """
    Train model on clients collection
    Example: Predict client category or status
    """
    print("\n" + "="*60)
    print("TRAINING MODEL ON CLIENTS COLLECTION")
    print("="*60 + "\n")
    
    # Update these based on your actual data structure
    COLLECTION = "clients"
    TARGET_COLUMN = "status"  # UPDATE THIS - e.g., 'category', 'type', 'active'
    TASK_TYPE = "classification"  # or "regression" if predicting numeric value
    DROP_COLUMNS = ['_id']  # Add any columns you want to exclude
    
    # Initialize predictor
    predictor = MongoDBCNNPredictor(
        mongo_uri=MONGO_URI,
        database_name=DATABASE,
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    # Connect and fetch data
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    print(f"Fetching data from {COLLECTION}...")
    df = predictor.fetch_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check if target column exists
    if TARGET_COLUMN not in df.columns:
        print(f"\n⚠ Warning: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease update TARGET_COLUMN in this script.")
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        df, 
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS
    )
    
    # Build model
    print("\nBuilding CNN model...")
    if TASK_TYPE == 'classification':
        num_classes = len(predictor.label_encoder.classes_)
        print(f"Number of classes: {num_classes}")
        predictor.build_cnn_model(
            input_shape=(X_train.shape[1], 1),
            num_classes=num_classes
        )
    else:
        predictor.build_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Train model
    print("\nTraining model...")
    predictor.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=8)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate(X_test, y_test)
    
    # Get insights
    print("\nGenerating insights...")
    insights = predictor.get_insights(X_test, y_test)
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save model
    print("\nSaving model...")
    predictor.save_model(
        model_path='clients_cnn_model.h5',
        scaler_path='clients_scaler.pkl'
    )
    
    print("\n✓ Training complete!")
    print(f"✓ Model saved as 'clients_cnn_model.h5'")

def train_commandes_model():
    """
    Train model on commandes collection
    Example: Predict order amount or status
    """
    print("\n" + "="*60)
    print("TRAINING MODEL ON COMMANDES COLLECTION")
    print("="*60 + "\n")
    
    # Update these based on your actual data structure
    COLLECTION = "commandes"
    TARGET_COLUMN = "montant"  # UPDATE THIS - e.g., 'total', 'amount', 'status'
    TASK_TYPE = "regression"  # or "classification" if predicting categories
    DROP_COLUMNS = ['_id']
    
    predictor = MongoDBCNNPredictor(
        mongo_uri=MONGO_URI,
        database_name=DATABASE,
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    print(f"Fetching data from {COLLECTION}...")
    df = predictor.fetch_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    if TARGET_COLUMN not in df.columns:
        print(f"\n⚠ Warning: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease update TARGET_COLUMN in this script.")
        return
    
    # Preprocess
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        df, 
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS
    )
    
    # Build model
    print("\nBuilding CNN model...")
    predictor.build_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Train
    print("\nTraining model...")
    predictor.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=8)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate(X_test, y_test)
    
    # Insights
    print("\nGenerating insights...")
    insights = predictor.get_insights(X_test, y_test)
    
    # Plot
    predictor.plot_training_history()
    
    # Save
    print("\nSaving model...")
    predictor.save_model(
        model_path='commandes_cnn_model.h5',
        scaler_path='commandes_scaler.pkl'
    )
    
    print("\n✓ Training complete!")
    print(f"✓ Model saved as 'commandes_cnn_model.h5'")

def train_produits_model():
    """
    Train model on produits collection
    Example: Predict product price or category
    """
    print("\n" + "="*60)
    print("TRAINING MODEL ON PRODUITS COLLECTION")
    print("="*60 + "\n")
    
    # Update these based on your actual data structure
    COLLECTION = "produits"
    TARGET_COLUMN = "prix"  # UPDATE THIS - e.g., 'price', 'category'
    TASK_TYPE = "regression"  # or "classification"
    DROP_COLUMNS = ['_id']
    
    predictor = MongoDBCNNPredictor(
        mongo_uri=MONGO_URI,
        database_name=DATABASE,
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    print(f"Fetching data from {COLLECTION}...")
    df = predictor.fetch_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    if TARGET_COLUMN not in df.columns:
        print(f"\n⚠ Warning: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease update TARGET_COLUMN in this script.")
        return
    
    # Preprocess
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = predictor.preprocess_data(
        df, 
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS
    )
    
    # Build model
    print("\nBuilding CNN model...")
    predictor.build_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Train
    print("\nTraining model...")
    predictor.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=8)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate(X_test, y_test)
    
    # Insights
    print("\nGenerating insights...")
    insights = predictor.get_insights(X_test, y_test)
    
    # Plot
    predictor.plot_training_history()
    
    # Save
    print("\nSaving model...")
    predictor.save_model(
        model_path='produits_cnn_model.h5',
        scaler_path='produits_scaler.pkl'
    )
    
    print("\n✓ Training complete!")
    print(f"✓ Model saved as 'produits_cnn_model.h5'")

def main():
    """Main function with menu"""
    print("\n" + "="*60)
    print("SHOPCAM CNN MODEL TRAINER")
    print("="*60)
    
    print("\nAvailable Collections:")
    print("1. clients   (10 documents)")
    print("2. commandes (15 documents)")
    print("3. produits  (? documents)")
    print("4. Train all collections")
    print("5. Exit")
    
    choice = input("\nSelect collection to train (1-5): ").strip()
    
    if choice == '1':
        train_clients_model()
    elif choice == '2':
        train_commandes_model()
    elif choice == '3':
        train_produits_model()
    elif choice == '4':
        print("\n🚀 Training models for all collections...\n")
        train_clients_model()
        train_commandes_model()
        train_produits_model()
    elif choice == '5':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the generated plots (training_history.png, etc.)")
    print("2. Use predict.py to make predictions")
    print("3. Load the saved model for production use")

if __name__ == "__main__":
    # First, recommend exploring the data
    print("\n💡 RECOMMENDATION:")
    print("Before training, run: python explore_shopcam_data.py")
    print("This will help you understand your data structure.\n")
    
    proceed = input("Continue with training? (y/n): ").strip().lower()
    
    if proceed == 'y':
        main()
    else:
        print("\n👉 Run: python explore_shopcam_data.py")
        print("Then come back to train your model!")
