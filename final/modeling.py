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
- Model Evaluation Functions
- Model Training Functions
- Misc/Helper Functions
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

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Timing
from time import time

# Matrix Manipulation
from scipy.sparse.linalg import eigsh

# ----- Global Variables --------------------------------------------------------------------------
# NA

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


def get_nn_model(num_features_to_train:int) -> models.Sequential:
    '''
    About
    -----
    - Creates and returns a simple forward-pass Neural Network for model training
    - This definition is to closely resemble the RandomForestClassifer as much as possible via the forward-pass

    Parameters
    ----------
    - num_features_to_train (int) :
        - The number of features being used from the dataset to train the NN on

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
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),

        # Output layer (using Sigmoid for binary or Softmax for multiclass)
        layers.Dense(1, activation='sigmoid') 
    ])
    
    # ----- Define Backpropagation Methodology ----------------------------------------------------
    nn_model.compile(
        optimizer='adam',                       # How weights/biases work
        loss='binary_crossentropy',             # How significant was the incorrectness
        metrics=['accuracy',                    # The metrics to optimize
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    return nn_model

# =================================================================================================
# END Model Definition Functions
# START Model Evaluation Functions
# =================================================================================================



# =================================================================================================
# END Model Evaluation Functions
# START Model Training Functions
# =================================================================================================



# =================================================================================================
# END Model Training Functions
# START Misc/Helper Functions
# =================================================================================================



# =================================================================================================
# END Misc/Helper Functions
# =================================================================================================