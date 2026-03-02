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
- Visualization Functions
    - vis_target_dist
    - vis_numerical
    - vis_cramer
    - vis_mutual_information
- Miscellaneous Helper Functions
    - _get_example_data
    - _get_markdown_data_dictionary
'''
# ----- Imports -----------------------------------------------------------------------------------
# Pathing and renaming files
import os

# Databasing
import numpy as np
import pandas as pd

# Reducing dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

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
# START Visualization Functions
# =================================================================================================

def vis_target_distribution(pdDataFrame:pd.DataFrame, target:str='attack') -> None:
    '''
    About
    -----
    - Returns a horizontal bar distribution of the target variable

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Pandas Dataframe to visualize distribution off of

    - target (str) :
        - Name of the target feature to visualize the distribution of

    Returns
    -------
    - Horizontal bar visualization of the target variable
    '''
    # Calculate counts and percentages
    counts = pdDataFrame[target].value_counts()
    pcts = (counts / len(pdDataFrame) * 100).round(4)

    # Create labels: "[Attack Name] ([Percentage]%)"
    labels = [f"{idx} ({p}%)" for idx, p in zip(counts.index, pcts)]

    # Base figure and bar design
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x=counts.values, y=labels, palette='viridis')

    # Logarithmic scale to show all values
    ax.set_xscale("log")

    # Descriptive labels
    plt.title(f'Distribution of 21 {target} Categories (Log Scale)')
    plt.xlabel('Log Count of Occurrences')
    plt.ylabel(f'{target} Type (Percentage of Total)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


def vis_numerical(pdDataFrame:pd.DataFrame, col_for_comparison:str, target:str='attack') -> None:
    '''
    About
    -----
    - Visualizes a violin plot of numerical data where the top 2, bottom 2, and middle most distributed attack types are displayed

    Parameters
    ----------
    - 

    Raises
    ------
    - 

    Returns
    -------
    - 
    '''
    # Determine distribution of target
    counts = pdDataFrame[target].value_counts()
    total = len(pdDataFrame)
    percentages = (counts / total * 100).round(2).to_dict()

    # Define the top, middle, and most rare targets
    top_2 = counts.index[:2]
    middle = [counts.index[len(counts)//2]]
    bottom_2 = counts.index[-2:]

    # Create a sub_dataframe to expedite lookup
    target_list = list(top_2) + list(middle) + list(bottom_2)
    plot_df = pdDataFrame[pdDataFrame[target].isin(target_list)].copy()

    # Create target label and distribution percentage
    plot_df['Attack Type'] = plot_df[target].apply(lambda x: f"{x} ({percentages[x]}%)")

    # Plot the violinplot
    plt.figure(figsize=(10, 8))
    sns.violinplot(
        data=plot_df, 
        x=col_for_comparison, 
        y='Attack Type', 
        log_scale=True, 
        hue=target,
        legend=False
    )
    plt.title(f'Distribution Snapshot of {col_for_comparison}: Top, Middle, and Rare {target}')
    plt.tight_layout()
    plt.show()

# =================================================================================================
# START Visualization Functions
# END Miscellaneous Helper Functions
# =================================================================================================

def _get_example_data(col_data:pd.Series, num_examples:int=3, is_target:bool=False) -> str:
    '''
    About
    -----
    - Returns "num_examples" from "col_data", if "is_target" returns all examples

    Parameters
    ----------
    - col_data (pd.Series) :
        - The Pandas column data to extract examples from

    - num_examples (int) :
        - Default: 3
        - Number of unique examples to return from col_data

    - is_target (bool) :
        - Default: Faluse
        - If true, will return all unique examples

    Returns
    -------
    - A string containing unique examples from "col_data"
    '''
    unique_vals = col_data.unique()

    # If target, return all unique_vals
    if is_target:
        return ", ".join(map(str, unique_vals))
    
    # Else, return up to num_examples in unique_vals
    examples = unique_vals[:num_examples]
    return ", ".join(map(str, examples))


def _get_markdown_data_dictionary(pdDataFrame:pd.DataFrame, num_examples:int=3, target:str='attack') -> None:
    '''
    About
    -----
    - Prints out a markdown friendly data dictionary that is formatted using "pdDataFrame", but needs
      the user to fill in the description.

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas Dataframe to format the majority of the markdown data dictionary off of

    - num_examples (int) :
        - Default: 3
        - Number of unique examples to return from col_data

    - target (str) :
        - The target column name to extract all unique values for examples

    Returns
    -------
    - Prints off the markdown friendly data dictionary as a string output statement
    '''
    # Header information
    print('| Feature Name | Dtype | Example | Description |')
    print('| :--- | :--- | :--- | :--- |')

    # Iterative data print
    for column in pdDataFrame.columns:
        datatype = pdDataFrame[column].dtype
        is_target_col = (column == target)
        example = _get_example_data(pdDataFrame[column], num_examples, is_target_col)
        print(f'| **{column}** | `{datatype}` | {example} | DESC |')

# =================================================================================================
# END Miscellaneous Helper Functions
# =================================================================================================