'''
FILE OVERVIEW:
- Underlying code intended for use in modeling.ipynb to keep notebook cleaner
- Consists of various functions intended to define ML/NN models, train ML/NN models, and evaluate ML/NN models

=================================================

MISC COMMENTS:
- NA

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Model Definition Functions
    - get_ml_model
    - get_nn_model
    - prepare_data_for_training
- Model Training Functions
    - train_ml
    - train_nn
- Misc/Helper Functions
    - set_reproducibility
'''
# ----- Imports -----------------------------------------------------------------------------------
# File Detection
import os

# Databasing
import numpy as np
import pandas as pd

# Networking
import networkx as nx
import igraph as ig

# Database splitting, encoding, scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
import random

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Timing
from time import time

# Matrix Manipulation
from scipy.sparse.linalg import eigsh

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# ----- Global Variables --------------------------------------------------------------------------
ATTACK_MAPPING = {'scanning': 16,
                  'benign': 2,
                  'ddos': 5,
                  'dos': 6,
                  'xss': 20,
                  'reconnaissance': 15,
                  'password': 13,
                  'injection': 11,
                  'brute_force': 4,
                  'fuzzers': 8,
                  'bot': 3,
                  'infilteration': 10,
                  'generic': 9,
                  'backdoor': 1,
                  'exploits': 7,
                  'ransomware': 14,
                  'mitm': 12,
                  'theft': 18,
                  'shellcode': 17,
                  'analysis': 0,
                  'worms': 19
                }

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Model Definition Functions
# =================================================================================================

def get_ml_model(trees:int=100) -> RandomForestClassifier:
    '''
    About
    -----
    - Creates and returns a basic sklearn RandomForestClassifier for model training

    Parameters
    ----------
    - trees (int) :
        - Default: 100
        - The number of decision trees to be created where each tree is randomly trained on a subset of the data.
          Essentially, this is like creating a voting block of whether or not something is significant during decisions

    Returns
    -------
    - RandomForestClassifier
        - An sklearn.ensemble RandomForesetClassifier ML model
    '''
    rfc_model = RandomForestClassifier(
        n_estimators=trees,      # This is just the number of "trees" we are creating
        class_weight='balanced', # This generally solves the imbalance issue
        max_features='sqrt',     # This generally solves the bias issue
        random_state=3703        # This is to ensure reproducibility
    )
    return rfc_model


def get_nn_model(num_features_to_train:int,
                 num_targets:int=21) -> models.Sequential:
    '''
    About
    -----
    - Creates and returns a simple forward-pass Neural Network for model training
    - This definition is to closely resemble the RandomForestClassifer as much as possible via the forward-pass

    Parameters
    ----------
    - num_features_to_train (int) :
        - The number of features being used from the dataset to train the NN on
    - num_targets (int) :
        - The number of categorical targets to predict

    Returns
    -------
    - models.Sequential
        - A tensorflow.keras NN model
    '''
    # ----- Define NN Structure -------------------------------------------------------------------
    nn_model = models.Sequential([

        # Input layer (Starting point of decision making)
        layers.Input(shape=(num_features_to_train,)),

        # Small hidden layers to prevent "brute force" memorization
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),

        # Output layer (using softmax for multiclass)
        layers.Dense(num_targets, activation='softmax') 
    ])
    
    # ----- Define Backpropagation Methodology ----------------------------------------------------
    opt = Adam(learning_rate=0.0005)
    nn_model.compile(
        optimizer=opt,                          # How weights/biases work
        loss='sparse_categorical_crossentropy', # How significant was the incorrectness
        metrics=['accuracy',                    # The metrics to optimize
                 tf.keras.metrics.SparseCategoricalAccuracy(name='cat_acc')
        ]
    )

    return nn_model


def prepare_data_for_training(data_for_training:pd.DataFrame, 
                              target:str = 'target',
                              cols_to_drop:list[str]=['attack', 'target']) -> dict:
    '''
    About
    -----
    - Prepares the normal dataset for model training
    - Returns train/val/test datasets, target weights to account for data imbalance, and the general shape of training and target variables

    Parameters
    ----------
    - data_for_training (pd.DataFrame) :
        - The Pandas dataframe representing the normal dataset to train models on (NOT COMPLEX NETWORK INFORMATION)
    - target (str) :
        - Default: target
        - The name of the target column to classify
    - cols_to_drop (list[str]) :
        - Default: [attack, target]
        - The name of the columns to drop (Ones that are not intended for training)

    Returns
    -------
    - dict
        - A dictionary housing variables necessary for model training
        - X_train: 70% of feature training data
        - X_val: 20% of feature validation data
        - X_test: 10% of feature testing data
        - y_train: 70% of target training data
        - y_val: 20% of target validation data
        - y_test: 10% of target testing data
        - class_weights: The weighted dictionary of the target variable to account for imbalanced data
        - num_features: Number of features to train on
        - num_targets: Number of target labels to identify
    '''
    # --- SAFETY CHECK: Handle rare classes for Stratification ---
    counts = data_for_training['target'].value_counts()
    rare_classes = counts[counts < 10].index
    
    if not rare_classes.empty:
        print(f"\033[33mWarning: Rare classes detected {list(rare_classes)}. Oversampling for stratification safety...\033[0m")
        for cls in rare_classes:
            # Duplicate the samples until we have at least 10
            samples = data_for_training[data_for_training['target'] == cls]
            multiplier = (10 // len(samples)) + 1
            oversampled = pd.concat([samples] * multiplier).iloc[:10]
            data_for_training = pd.concat([data_for_training[data_for_training['target'] != cls], oversampled])
            
    # Define training features (X) and target feature (y)
    X = data_for_training.drop(columns=cols_to_drop)
    y = data_for_training[target] # The 0-20 integer IDs

    # Split into train, val, test sets (70-20-10 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=3703, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.33, random_state=3703, stratify=y_val
    )

    # Calculate attack type weights to catch imbalanced data
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))

    # Define the number of training features and target features
    num_features = X_train.shape[1]
    num_targets = 21

    # Prepare and return dict of information
    preparation_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_weights": class_weights_dict,
        "num_features": num_features,
        "num_targets": num_targets
    }
    return preparation_dict

# =================================================================================================
# END Model Definition Functions
# START Model Training Functions
# =================================================================================================

def train_ml(preparation_dict:dict,
             model_save_name:str = 'models/ml_normal_model.joblib',
             metrics_save_name:str = 'metrics/ml_normal_metrics.parquet') -> pd.DataFrame:
    '''
    About
    -----
    - Conducts Random Forest ML model training and saves the trained model
    - Conducts model evaluation and metrics saving which is returned as a Pandas dataframe for immediate use

    Parameters
    ----------
    - preparation_dict (dict) :
        - The dictionary of the necessary normal dataset preparation ideally from the prepare_data_for_training() function
    - model_save_name (str) :
        - Default: models/ml_normal_model.joblib
        - The name of the trained Random Forest model save file
    - metrics_save_name (str) :
        - Default: metrics/ml_normal_metrics.parquet
        - The name of the Random Forest model performance metrics save file

    Returns
    -------
    - pd.DataFrame
        - The Random Forest performance metrics as a Pandas dataframe
    '''
    # ----- Extract Data From Preparation Dict ----------------------------------------------------
    X_train, y_train = preparation_dict['X_train'], preparation_dict['y_train']
    X_test, y_test  = preparation_dict['X_test'], preparation_dict['y_test']
    
    # ----- Train Model ---------------------------------------------------------------------------
    # Initialize and train model
    print(f'\033[33mTraining Random Forest on {X_train.shape[0]} rows...\033[0m')
    rf_model = get_ml_model()
    rf_model.fit(X_train, y_train)

    # Save model
    # This is the model, NOT the metrics
    joblib.dump(rf_model, model_save_name, compress=3)
    print(f'\033[32mModel successfully saved as {model_save_name}!\033[0m')

    # ----- Evaluate Model ------------------------------------------------------------------------
    print('\033[33mTesting Random Forest...\033[0m')
    rf_preds = rf_model.predict(X_test)
    print('\033[32mRandom Forest testing complete!\n\033[0m'
          '\033[33mObtaining performance metrics and saving...\033[0m')
    
    # Obtain the dictionary report for saving later
    report_dict = classification_report(y_test, rf_preds, output_dict=True)
    
    # Convert to DataFrame for readability and saving
    report_df = pd.DataFrame(report_dict).transpose()

    # Map the actual attack names to the attack ID
    report_df = report_df.reset_index().rename(columns={'index': 'target'})
    mapping_df = pd.DataFrame(list(ATTACK_MAPPING.items()), columns=['attack_name', 'target'])
    report_df['target'] = report_df['target'].astype(str)
    mapping_df['target'] = mapping_df['target'].astype(str)
    report_df = pd.merge(report_df, mapping_df, on='target', how='left')
    report_df['attack_name'] = report_df['attack_name'].fillna(report_df['target'])
    report_df = report_df.set_index('attack_name')
    
    # Save the Metrics
    report_df.to_parquet(metrics_save_name)
    print(f'\033[32mRandom Forest metrics successfully saved as {metrics_save_name}!\033[0m')

    # Return the metrics df if it is immediately desired
    return report_df


def train_nn(preparation_dict:dict,
             model_save_name:str = 'models/nn_normal_model.keras',
             history_save_name:str = 'metrics/nn_normal_history.parquet',
             metrics_save_name:str = 'metrics/nn_normal_metrics.parquet') -> pd.DataFrame:
    '''
    About
    -----
    - Conducts NN model training and saves the trained model
    - Conducts model evaluation and metrics saving which is returned as a Pandas dataframe for immediate use

    Parameters
    ----------
    - preparation_dict (dict) :
        - The dictionary of the necessary normal dataset preparation ideally from the prepare_data_for_training() function
    - model_save_name (str) :
        - Default: models/nn_normal_model.joblib
        - The name of the trained NN model save file
    - history_save_name (str) :
        - Default: metrics/nn_normal_history.parquet
        - The name of the training history of the NN model
    - metrics_save_name (str) :
        - Default: metrics/nn_normal_metrics.parquet
        - The name of the NN model performance metrics save file

    Returns
    -------
    - pd.DataFrame
        - The NN performance metrics as a Pandas dataframe
    '''
    # ----- Extract Data --------------------------------------------------------------------------
    X_train, y_train = preparation_dict['X_train'], preparation_dict['y_train']
    X_val, y_val     = preparation_dict['X_val'],   preparation_dict['y_val']
    X_test, y_test   = preparation_dict['X_test'],  preparation_dict['y_test']
    
    # ----- Initialize Model ----------------------------------------------------------------------
    nn_model = get_nn_model(num_features_to_train=preparation_dict['num_features'], 
                            num_targets=preparation_dict['num_targets'])

    # ----- Define Stop Switches ------------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # ----- Train Model ---------------------------------------------------------------------------
    print(f'\033[33mTraining Neural Network on {X_train.shape[0]} rows...\033[0m')
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), # THE VALIDATION PASS TO HELP PREVENT OVERFITTING
        epochs=50, 
        batch_size=2048,
        class_weight=preparation_dict['class_weights'],
        callbacks=callbacks,
        verbose=1
    )

    print('\033[32mNeural Network training completed!\033[0m')
    nn_model.save(model_save_name)
    print(f'\033[32mTrained Neural Network for normal data successfully saved as {model_save_name}!\033[0m')

    history_df = pd.DataFrame(history.history)
    history_df.to_parquet(history_save_name)
    print(f'\033[32mSuccessfully saved off training histroy as {history_save_name}!\033[0m')

    # ----- Evaluate Model ------------------------------------------------------------------------
    print(f'\033[33mTesting Neural Network...\033[0m')
    nn_probs = nn_model.predict(X_test, batch_size=4096)
    nn_preds = np.argmax(nn_probs, axis=1)
    print('\033[32mNeural Network testing complete!\n\033[0m'
          '\033[33mObtaining performance metrics and saving...\033[0m')
    
    # Metrics
    report_dict = classification_report(y_test, nn_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Map the actual attack names to the attack ID
    report_df = report_df.reset_index().rename(columns={'index': 'target'})
    mapping_df = pd.DataFrame(list(ATTACK_MAPPING.items()), columns=['attack_name', 'target'])
    report_df['target'] = report_df['target'].astype(str)
    mapping_df['target'] = mapping_df['target'].astype(str)
    report_df = pd.merge(report_df, mapping_df, on='target', how='left')
    report_df['attack_name'] = report_df['attack_name'].fillna(report_df['target'])
    report_df = report_df.set_index('attack_name')
    report_df.to_parquet(metrics_save_name)

    print(f'\033[32mNeural Network metrics successfully saved as {metrics_save_name}!\033[0m')
    
    # Return metrics
    return report_df

# =================================================================================================
# END Model Training Functions
# START Misc/Helper Functions
# =================================================================================================

def set_reproducibility(seed=3703) -> None:
    '''
    About
    -----
    - Establishes a random seed for the environment, tensorflow, and numpy to ensure reproducibility as much as possible

    Parameters
    ----------
    - seed (int) :
        - Default: 3703
        - The seed to be set in multiple areas for reproducibilit
    
    Returns
    -------
    - None, just sets the seed in several areas to ensure reproducibility
    '''
    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# =================================================================================================
# END Misc/Helper Functions
# =================================================================================================