"""
Configuration file for MongoDB CNN Predictor
Update these settings according to your MongoDB setup and requirements
"""

# MongoDB Configuration
MONGODB_CONFIG = {
    # MongoDB Atlas connection URI
    'uri': 'mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority',
    
    # Database name
    'database': 'shopcam',
    
    # Collection name (choose: 'clients', 'commandes', or 'produits')
    'collection': 'clients',
    
    # Query filter (optional)
    # Example: {'status': 'active', 'year': {'$gte': 2020}}
    'query': {},
    
    # Maximum number of documents to fetch (None for all)
    'limit': None
}

# Model Configuration
MODEL_CONFIG = {
    # Task type: 'classification' or 'regression'
    'task_type': 'classification',
    
    # Target column name in your dataset
    'target_column': 'target',
    
    # Columns to drop from the dataset (if any)
    'drop_columns': ['id', 'timestamp'],  # Example columns to drop
    
    # Training parameters
    'test_size': 0.2,  # Proportion of data for testing
    'random_state': 42,  # Random seed for reproducibility
    'epochs': 100,  # Maximum number of training epochs
    'batch_size': 32,  # Batch size for training
    
    # Early stopping patience
    'early_stopping_patience': 15,
    
    # Learning rate
    'learning_rate': 0.001
}

# CNN Architecture Configuration
CNN_ARCHITECTURE = {
    # First convolutional block
    'conv1_filters': 64,
    'conv1_kernel_size': 3,
    'conv1_pool_size': 2,
    'conv1_dropout': 0.3,
    
    # Second convolutional block
    'conv2_filters': 128,
    'conv2_kernel_size': 3,
    'conv2_pool_size': 2,
    'conv2_dropout': 0.3,
    
    # Third convolutional block
    'conv3_filters': 256,
    'conv3_kernel_size': 3,
    'conv3_dropout': 0.4,
    
    # Dense layers
    'dense1_units': 128,
    'dense1_dropout': 0.5,
    'dense2_units': 64,
    'dense2_dropout': 0.3
}

# File Paths
FILE_PATHS = {
    'model': 'cnn_model.h5',
    'scaler': 'scaler.pkl',
    'label_encoder': 'label_encoder.pkl',
    'metadata': 'model_metadata.json',
    'training_history_plot': 'training_history.png',
    'confusion_matrix_plot': 'confusion_matrix.png',
    'predictions_plot': 'predictions_vs_actual.png'
}

# Logging Configuration
LOGGING_CONFIG = {
    'verbose': True,  # Print detailed logs
    'save_plots': True,  # Save visualization plots
    'plot_dpi': 300  # DPI for saved plots
}

# Example configurations for common use cases

# Classification Example (shopcam clients analysis)
CLASSIFICATION_EXAMPLE = {
    'mongodb': {
        'uri': 'mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority',
        'database': 'shopcam',
        'collection': 'clients',
        'query': {},
        'limit': None
    },
    'model': {
        'task_type': 'classification',
        'target_column': 'status',  # Update based on your data
        'drop_columns': ['_id'],
        'epochs': 50,
        'batch_size': 32
    }
}

# Regression Example (shopcam commandes prediction)
REGRESSION_EXAMPLE = {
    'mongodb': {
        'uri': 'mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority',
        'database': 'shopcam',
        'collection': 'commandes',
        'query': {},
        'limit': None
    },
    'model': {
        'task_type': 'regression',
        'target_column': 'montant',  # Update based on your data
        'drop_columns': ['_id'],
        'epochs': 100,
        'batch_size': 32
    }
}

# Product Analysis Example (shopcam produits)
TIME_SERIES_EXAMPLE = {
    'mongodb': {
        'uri': 'mongodb+srv://lmuij113_db_user:e03bEYLbqCpcetGk@cluster0.y3swaud.mongodb.net/?retryWrites=true&w=majority',
        'database': 'shopcam',
        'collection': 'produits',
        'query': {},
        'limit': None
    },
    'model': {
        'task_type': 'regression',
        'target_column': 'prix',  # Update based on your data
        'drop_columns': ['_id'],
        'epochs': 100,
        'batch_size': 32
    }
}

# Helper function to get configuration
def get_config(config_type='default'):
    """
    Get configuration based on type
    
    Args:
        config_type (str): 'default', 'classification', 'regression', or 'timeseries'
        
    Returns:
        dict: Configuration dictionary
    """
    if config_type == 'classification':
        return CLASSIFICATION_EXAMPLE
    elif config_type == 'regression':
        return REGRESSION_EXAMPLE
    elif config_type == 'timeseries':
        return TIME_SERIES_EXAMPLE
    else:
        return {
            'mongodb': MONGODB_CONFIG,
            'model': MODEL_CONFIG,
            'architecture': CNN_ARCHITECTURE,
            'paths': FILE_PATHS,
            'logging': LOGGING_CONFIG
        }

# Validation function
def validate_config(config):
    """
    Validate configuration settings
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['mongodb', 'model']
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    # Validate task type
    if config['model']['task_type'] not in ['classification', 'regression']:
        print(f"Error: Invalid task_type. Must be 'classification' or 'regression'")
        return False
    
    # Validate MongoDB URI
    if not config['mongodb']['uri']:
        print(f"Error: MongoDB URI cannot be empty")
        return False
    
    print("✓ Configuration validated successfully")
    return True
