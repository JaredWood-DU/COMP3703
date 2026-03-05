'''
FILE OVERVIEW:
- Underlying code intended for use in explore_complex_networks.ipynb to keep notebook cleaner
- Consists of various functions intended to derive complex network information to engineer into the dataset

=================================================

MISC COMMENTS:
- NA

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Helper Functions
    - Thing 1
    - Thing 2
- Main Function
'''
# ----- Imports -----------------------------------------------------------------------------------
# Databasing
import numpy as np
import pandas as pd

# Networking
import networkx as nx
import igraph as ig

# Database splitting, encoding, scaling
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Global Variables --------------------------------------------------------------------------
# NA

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
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


def thing2():
    pass

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================



# =================================================================================================
# END Main Function
# =================================================================================================