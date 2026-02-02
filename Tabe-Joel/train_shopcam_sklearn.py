"""
Train ML Model on Shopcam Data (Scikit-learn version - Python 3.14 compatible)
Uses Random Forest and other sklearn models instead of CNN
Reads MongoDB connection from .env file
"""

from ml_mongodb_model import MongoDBMLPredictor
import sys
import os

def train_clients_model():
    """Train model on clients collection"""
    print("\n" + "="*60)
    print("TRAINING MODEL ON CLIENTS COLLECTION")
    print("="*60 + "\n")
    
    # Configuration
    COLLECTION = "clients"
    TARGET_COLUMN = "ville"  # Predict city (Douala or Yaoundé) based on age
    # Other options: "age" (regression)
    TASK_TYPE = "classification"  # or "regression"
    DROP_COLUMNS = ['_id', 'nom']  # Drop ID and name (not useful for prediction)
    MODEL_TYPE = "random_forest"  # Options: 'random_forest', 'gradient_boosting', 'xgboost'
    
    # Initialize predictor (reads MONGO_DB_STRING from .env)
    predictor = MongoDBMLPredictor(
        database_name="shopcam",
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    # Connect and fetch data
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB. Check your .env file.")
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
        print(f"\n⚠ ERROR: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        print("\n💡 Steps to fix:")
        print("1. Run: python explore_shopcam_data.py")
        print("2. Find the correct column name")
        print("3. Update TARGET_COLUMN in this script")
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    try:
        X_train, X_test, y_train, y_test = predictor.preprocess_data(
            df, 
            target_column=TARGET_COLUMN,
            drop_columns=DROP_COLUMNS
        )
    except Exception as e:
        print(f"✗ Preprocessing error: {str(e)}")
        return
    
    # Build model
    print(f"\nBuilding {MODEL_TYPE} model...")
    predictor.build_model(model_type=MODEL_TYPE)
    
    # Train model
    print("\nTraining model...")
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\nEvaluating model...")
    results = predictor.evaluate(X_test, y_test)
    
    # Feature importance
    predictor.get_feature_importance()
    
    # Plot results
    print("\nGenerating visualizations...")
    predictor.plot_results(y_test, results['predictions'])
    
    # Save model
    print("\nSaving model...")
    predictor.save_model(model_path='clients_ml_model.pkl')
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Model saved as 'clients_ml_model.pkl'")
    print(f"✓ Scaler saved as 'scaler.pkl'")
    print(f"✓ Plots saved as PNG files")


def train_commandes_model():
    """Train model on commandes collection"""
    print("\n" + "="*60)
    print("TRAINING MODEL ON COMMANDES COLLECTION")
    print("="*60 + "\n")
    
    COLLECTION = "commandes"
    TARGET_COLUMN = "quantite"  # Predict order quantity
    # Other options: "ville_livraison" (classification), "client_id" (classification)
    TASK_TYPE = "regression"  # or "classification"
    DROP_COLUMNS = ['_id']
    MODEL_TYPE = "random_forest"
    
    predictor = MongoDBMLPredictor(
        database_name="shopcam",
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    df = predictor.fetch_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    if TARGET_COLUMN not in df.columns:
        print(f"\n⚠ ERROR: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    try:
        X_train, X_test, y_train, y_test = predictor.preprocess_data(
            df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS
        )
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return
    
    predictor.build_model(model_type=MODEL_TYPE)
    predictor.train(X_train, y_train, X_test, y_test)
    results = predictor.evaluate(X_test, y_test)
    predictor.get_feature_importance()
    predictor.plot_results(y_test, results['predictions'])
    predictor.save_model(model_path='commandes_ml_model.pkl')
    
    print("\n✓ Training complete! Model saved as 'commandes_ml_model.pkl'")


def train_produits_model():
    """Train model on produits collection"""
    print("\n" + "="*60)
    print("TRAINING MODEL ON PRODUITS COLLECTION")
    print("="*60 + "\n")
    
    COLLECTION = "produits"
    TARGET_COLUMN = "prix"  # Predict product price
    # Other options: "nom" (classification)
    TASK_TYPE = "regression"
    DROP_COLUMNS = ['_id', 'nom']  # Drop ID and name
    MODEL_TYPE = "random_forest"
    
    predictor = MongoDBMLPredictor(
        database_name="shopcam",
        collection_name=COLLECTION,
        task_type=TASK_TYPE
    )
    
    if not predictor.connect_to_mongodb():
        print("Failed to connect to MongoDB")
        return
    
    df = predictor.fetch_data()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\n✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    if TARGET_COLUMN not in df.columns:
        print(f"\n⚠ ERROR: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    try:
        X_train, X_test, y_train, y_test = predictor.preprocess_data(
            df, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS
        )
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return
    
    predictor.build_model(model_type=MODEL_TYPE)
    predictor.train(X_train, y_train, X_test, y_test)
    results = predictor.evaluate(X_test, y_test)
    predictor.get_feature_importance()
    predictor.plot_results(y_test, results['predictions'])
    predictor.save_model(model_path='produits_ml_model.pkl')
    
    print("\n✓ Training complete! Model saved as 'produits_ml_model.pkl'")


def main():
    """Main function with menu"""
    # Check for .env file
    if not os.path.exists('.env'):
        print("\n⚠ ERROR: .env file not found!")
        print("Please create a .env file with:")
        print("MONGO_DB_STRING=your_mongodb_connection_string")
        return
    
    print("\n" + "="*60)
    print("SHOPCAM ML MODEL TRAINER (Scikit-learn)")
    print("Python 3.14 Compatible - No TensorFlow required")
    print("="*60)
    
    print("\nAvailable Collections:")
    print("1. clients   (10 documents)")
    print("2. commandes (15 documents)")
    print("3. produits")
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


if __name__ == "__main__":
    print("\n💡 RECOMMENDATION:")
    print("First run: python explore_shopcam_data.py")
    print("This shows you all columns in your collections.\n")
    
    proceed = input("Continue with training? (y/n): ").strip().lower()
    
    if proceed == 'y':
        main()
    else:
        print("\n👉 Run: python explore_shopcam_data.py")
