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
    - thing1
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

def thing1():
    '''
    About
    -----
    - Some placeholder function

    Parameters
    ----------
    - ray_nn_train_func (Function) :
        - The Ray-Train function logic for MLflow to wrap and log information from

    - framework (str) :
        - Default: pytorch (Not implemented)
        - String representation of the NN framework used (NOT IMPLEMENTED)

    Raises
    ------
    - RunTimeError
        - Generally if anything should fail to log properly

    - NotImplementedError
        - Generally if something has not been implemented yet, particularly with framework types

    Returns
    -------
    - Wraps the Ray-train function with MLflow logging logic to display results on MLflow UI
    '''
    pass

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