"""
Explore Shopcam Database (Updated - Uses .env)
Quick script to view and understand your MongoDB data
"""

from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection from .env
MONGO_URI = os.getenv('MONGO_DB_STRING')
DATABASE = "shopcam"

if not MONGO_URI:
    print("ERROR: MONGO_DB_STRING not found in .env file")
    print("Please create a .env file with your MongoDB connection string")
    exit(1)

def explore_collection(collection_name):
    """Explore a specific collection"""
    print(f"\n{'='*60}")
    print(f"COLLECTION: {collection_name}")
    print('='*60)
    
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE]
        collection = db[collection_name]
        
        # Count documents
        count = collection.count_documents({})
        print(f"\n✓ Total documents: {count}")
        
        if count == 0:
            print("⚠ Collection is empty")
            client.close()
            return
        
        # Get sample documents
        sample = list(collection.find().limit(min(5, count)))
        
        if sample:
            # Convert to DataFrame for better display
            df = pd.DataFrame(sample)
            
            print(f"\n📊 Columns ({len(df.columns)}): {list(df.columns)}")
            print(f"\n📊 Data types:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            print(f"\n📋 Sample data (first {min(5, count)} records):")
            print(df.head().to_string())
            
            # Numeric columns stats
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"\n📈 Numeric column statistics:")
                print(df[numeric_cols].describe())
            
            # Categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                print(f"\n📊 Categorical columns unique values:")
                for col in cat_cols:
                    unique_count = df[col].nunique()
                    print(f"  {col}: {unique_count} unique values")
                    if unique_count <= 10:
                        print(f"    Values: {df[col].unique().tolist()}")
            
            print(f"\n🔍 Missing values:")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0])
            else:
                print("  No missing values ✓")
            
        client.close()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def main():
    """Main function to explore all collections"""
    print("\n" + "="*60)
    print("SHOPCAM DATABASE EXPLORER")
    print("="*60)
    print(f"Connected to: {DATABASE}")
    
    collections = ['clients', 'commandes', 'produits']
    
    for collection_name in collections:
        explore_collection(collection_name)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    
    # Provide suggestions
    print("\n💡 Next Steps:")
    print("1. Review the data structure above")
    print("2. Identify your target column for prediction")
    print("3. Update target_column in train_shopcam_sklearn.py")
    print("4. Run: python train_shopcam_sklearn.py")

if __name__ == "__main__":
    main()
