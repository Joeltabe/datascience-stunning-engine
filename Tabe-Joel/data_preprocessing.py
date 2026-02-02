"""
Data Preprocessing Module for MongoDB CNN Predictor
Handles data fetching, cleaning, and transformation
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MongoDBDataPreprocessor:
    """
    Handles data preprocessing for MongoDB data before CNN training
    """
    
    def __init__(self, mongo_uri, database_name, collection_name):
        """
        Initialize preprocessor
        
        Args:
            mongo_uri (str): MongoDB connection URI
            database_name (str): Database name
            collection_name (str): Collection name
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.scalers = {}
        self.encoders = {}
        
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            print(f"✓ Connected to MongoDB: {self.database_name}.{self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")
    
    def fetch_data(self, query={}, projection=None, limit=None, sort_by=None):
        """
        Fetch data from MongoDB with advanced options
        
        Args:
            query (dict): MongoDB query filter
            projection (dict): Fields to include/exclude
            limit (int): Maximum documents to fetch
            sort_by (list): Sort criteria [(field, direction)]
            
        Returns:
            pd.DataFrame: Fetched data
        """
        try:
            cursor = self.collection.find(query, projection)
            
            if sort_by:
                cursor = cursor.sort(sort_by)
            
            if limit:
                cursor = cursor.limit(limit)
            
            data = list(cursor)
            df = pd.DataFrame(data)
            
            # Remove MongoDB _id if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            print(f"✓ Fetched {len(df)} records")
            print(f"✓ Columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            print(f"✗ Fetch error: {str(e)}")
            return pd.DataFrame()
    
    def get_data_info(self, df):
        """
        Display comprehensive data information
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        print("\n=== Data Information ===")
        print(f"Shape: {df.shape}")
        print(f"\nColumns and Types:")
        print(df.dtypes)
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(df.describe())
        print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def handle_missing_values(self, df, strategy='mean', categorical_strategy='most_frequent'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for numeric columns ('mean', 'median', 'constant')
            categorical_strategy (str): Strategy for categorical columns
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        print("\n--- Handling Missing Values ---")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            print(f"✓ Imputed {len(numeric_cols)} numeric columns with strategy: {strategy}")
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            print(f"✓ Imputed {len(categorical_cols)} categorical columns with strategy: {categorical_strategy}")
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers from specified columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers (None = all numeric)
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe without outliers
        """
        print(f"\n--- Removing Outliers ({method.upper()} method) ---")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        original_size = len(df)
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        removed = original_size - len(df)
        print(f"✓ Removed {removed} outlier rows ({removed/original_size*100:.2f}%)")
        
        return df
    
    def encode_categorical(self, df, columns=None, method='onehot'):
        """
        Encode categorical variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to encode (None = all categorical)
            method (str): 'onehot' or 'label'
            
        Returns:
            pd.DataFrame: Encoded dataframe
        """
        print(f"\n--- Encoding Categorical Variables ({method}) ---")
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            print(f"✓ One-hot encoded {len(columns)} columns")
        
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            print(f"✓ Label encoded {len(columns)} columns")
        
        return df
    
    def scale_features(self, df, columns=None, method='standard'):
        """
        Scale numeric features
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to scale (None = all numeric)
            method (str): 'standard' or 'minmax'
            
        Returns:
            pd.DataFrame: Scaled dataframe
        """
        print(f"\n--- Scaling Features ({method}) ---")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"Unknown scaling method: {method}")
            return df
        
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers['feature_scaler'] = scaler
        
        print(f"✓ Scaled {len(columns)} columns")
        return df
    
    def create_time_features(self, df, datetime_column):
        """
        Extract time-based features from datetime column
        
        Args:
            df (pd.DataFrame): Input dataframe
            datetime_column (str): Name of datetime column
            
        Returns:
            pd.DataFrame: Dataframe with time features
        """
        print(f"\n--- Creating Time Features from '{datetime_column}' ---")
        
        # Convert to datetime if not already
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # Extract features
        df['year'] = df[datetime_column].dt.year
        df['month'] = df[datetime_column].dt.month
        df['day'] = df[datetime_column].dt.day
        df['dayofweek'] = df[datetime_column].dt.dayofweek
        df['quarter'] = df[datetime_column].dt.quarter
        df['weekofyear'] = df[datetime_column].dt.isocalendar().week
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Drop original datetime column
        df = df.drop(datetime_column, axis=1)
        
        print("✓ Created time features: year, month, day, dayofweek, quarter, weekofyear, is_weekend")
        
        return df
    
    def balance_dataset(self, df, target_column, method='undersample'):
        """
        Balance imbalanced dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column name
            method (str): 'undersample', 'oversample', or 'smote'
            
        Returns:
            pd.DataFrame: Balanced dataframe
        """
        print(f"\n--- Balancing Dataset ({method}) ---")
        
        from collections import Counter
        
        print(f"Original class distribution: {Counter(df[target_column])}")
        
        if method == 'undersample':
            # Undersample majority class
            min_class_count = df[target_column].value_counts().min()
            df_balanced = df.groupby(target_column).sample(n=min_class_count, random_state=42)
        
        elif method == 'oversample':
            # Oversample minority class
            max_class_count = df[target_column].value_counts().max()
            df_balanced = df.groupby(target_column).sample(n=max_class_count, replace=True, random_state=42)
        
        elif method == 'smote':
            from imblearn.over_sampling import SMOTE
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            df_balanced = pd.concat([X_balanced, y_balanced], axis=1)
        
        else:
            print(f"Unknown balancing method: {method}")
            return df
        
        print(f"Balanced class distribution: {Counter(df_balanced[target_column])}")
        
        return df_balanced
    
    def feature_engineering(self, df):
        """
        Perform custom feature engineering
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Enhanced dataframe
        """
        print("\n--- Feature Engineering ---")
        
        # Example: Create interaction features
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]  # Limit to first 3
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        print(f"✓ Created interaction features")
        
        return df
    
    def prepare_for_cnn(self, X):
        """
        Reshape data for CNN input (samples, timesteps, features)
        
        Args:
            X (array): Input features
            
        Returns:
            array: Reshaped data
        """
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"✓ Reshaped data for CNN: {X.shape}")
        return X
    
    def full_preprocessing_pipeline(self, query={}, target_column='target', 
                                   drop_columns=[], datetime_column=None,
                                   handle_outliers=True, balance_data=False):
        """
        Complete preprocessing pipeline
        
        Args:
            query (dict): MongoDB query
            target_column (str): Target variable name
            drop_columns (list): Columns to drop
            datetime_column (str): Datetime column name
            handle_outliers (bool): Whether to remove outliers
            balance_data (bool): Whether to balance classes
            
        Returns:
            tuple: (X, y, feature_names)
        """
        print("\n" + "="*50)
        print("FULL PREPROCESSING PIPELINE")
        print("="*50)
        
        # 1. Fetch data
        df = self.fetch_data(query)
        
        if df.empty:
            print("✗ No data fetched")
            return None, None, None
        
        # 2. Display info
        self.get_data_info(df)
        
        # 3. Drop unnecessary columns
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
            print(f"\n✓ Dropped columns: {drop_columns}")
        
        # 4. Handle datetime features
        if datetime_column and datetime_column in df.columns:
            df = self.create_time_features(df, datetime_column)
        
        # 5. Handle missing values
        df = self.handle_missing_values(df)
        
        # 6. Remove outliers
        if handle_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != target_column]
            if len(numeric_cols) > 0:
                df = self.remove_outliers(df, columns=numeric_cols)
        
        # 7. Separate features and target
        if target_column not in df.columns:
            print(f"✗ Target column '{target_column}' not found")
            return None, None, None
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 8. Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X = self.encode_categorical(X, columns=categorical_cols, method='onehot')
        
        # 9. Balance dataset if classification
        if balance_data and y.dtype == 'object' or y.nunique() < 20:
            df_temp = pd.concat([X, y], axis=1)
            df_temp = self.balance_dataset(df_temp, target_column)
            X = df_temp.drop(columns=[target_column])
            y = df_temp[target_column]
        
        # 10. Get feature names
        feature_names = X.columns.tolist()
        
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Final shape - Features: {X.shape}, Target: {y.shape}")
        print(f"✓ Total features: {len(feature_names)}")
        
        return X.values, y.values, feature_names


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MongoDBDataPreprocessor(
        mongo_uri="mongodb://localhost:27017/",
        database_name="your_database",
        collection_name="your_collection"
    )
    
    # Connect to MongoDB
    if preprocessor.connect():
        # Run full pipeline
        X, y, feature_names = preprocessor.full_preprocessing_pipeline(
            query={},
            target_column='target',
            drop_columns=['id'],
            handle_outliers=True,
            balance_data=False
        )
        
        print(f"\nFeatures ready for CNN training!")
        print(f"Shape: X={X.shape}, y={y.shape}")
        
        # Disconnect
        preprocessor.disconnect()
