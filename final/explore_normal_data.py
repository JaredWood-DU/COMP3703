'''
FILE OVERVIEW:
- Underlying code intended for use in explore_normal_data.ipynb to keep notebook cleaner
- Consists of function that determine the statistical significance and the associative strength
  of a feature in relation to the target feature

=================================================

MISC COMMENTS:
- The explore_normal_data.ipynb explains in better detail the statistical application, but to reitterate:
    - I am NOT a cybersecurity SME
    - I WILL RELY HEAVILY on statistical analysis to determine what features are used for model training

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Statistical Analysis Functions
    - analyze_chi_square
    - analyze_kruskal_wallis
- Associativity Functions
    - analyze_cramers
    - analyze_mutual_information
- Dual-Purpose Function
    - analyze_statical_significance_and_associativity
'''
# ----- Imports -----------------------------------------------------------------------------------
# Pathing and renaming files
import os

# Databasing
import numpy as np
import pandas as pd

# Reducing dataset
from sklearn.model_selection import train_test_split

# ----- Global Variables --------------------------------------------------------------------------


# =================================================================================================
# END File Overview, Imports, Global Variables
# START Statistical Analysis Functions
# =================================================================================================

def analyze_chi_square():
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


def analyze_kruskal_wallis():
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
# END Statistical Analysis Functions
# START Associativity Functions
# =================================================================================================

def analyze_cramers():
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


def analyze_mutual_information():
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
# END Associativity Functions
# START Dual-Purpose Function
# =================================================================================================

def analyze_statistical_significance_and_associativity():
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
# END Dual-Purpose Function
# =================================================================================================