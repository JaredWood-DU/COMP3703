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
    - vis_categorical
    - vis_singular_association
- Preprocess Functions
    - preprocess_normal_data
- Miscellaneous Helper Functions
    - _get_example_data
    - _get_markdown_data_dictionary
'''
# ----- Imports -----------------------------------------------------------------------------------
# Databasing
import numpy as np
import pandas as pd

# Database splitting, encoding, scaling
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# Statistical testing
from scipy.stats import kruskal, chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Global Variables --------------------------------------------------------------------------
# NA

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Statistical Analysis Functions
# =================================================================================================

def analyze_chi_square(pdDataFrame:pd.DataFrame,
                       feature:str,
                       target:str='attack',
                       p_threshold:float=0.05) -> dict:
    '''
    About
    -----
    - Applies scipy's chi2_contingency on a categorical feature and prints a color-coded statistic and p_value
      where green is the p_value is less than p_threshold, red otherwise
    - Returns a dictionary of all the information derived from the chi2_contingency function
    - This is intended to statistically analyze CATEGORICAL features

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Pandas dataframe to apply the statistic test on

    - feature (str) :
        - The name of the feature (column) to be tested with "target"
    
    - target (str) :
        - Default: attack
        - The name of the target feature (column) to be tested with "feature"

    - p_threshold (float) :
        - Default: 0.05
        - The value to determine if the statistic is statistically significant or not (p_val <= p_threshold means statistically significant)

    Returns
    -------
    - A color-coded statistic and p_value where green means p_val <= p_threshold using scipy's chi2_contingency as a print statement
    - A dictionary holding statistical test information
    '''
    # Setup and conduct test
    contingency_table = pd.crosstab(pdDataFrame[feature], pdDataFrame[target])
    stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    # Create dictionary of outputs
    stat_results = {
        'contingency_table': contingency_table,
        'feature': feature,
        'target': target,
        'stat': stat,
        'p_val': p_val,
        'dof': dof,
        'expected': expected
    }

    # Print off color-coded results
    if p_val <= p_threshold:
        print(f'\033[32m{feature} IS statistically significant with {target}!\033[0m\n'
              f'\033[35mStatistic:\033[0m {stat:.4f}\n'
              f'\033[35mP-Val:\033[0m {p_val:.4f}\n'
              f'\033[35mP-Threshold:\033[0m {p_threshold:.4f}')
    else:
        print(f'\033[33m{feature} IS NOT statistically significant with {target}!\033[0m\n'
              f'\033[35mStatistic:\033[0m {stat:.4f}\n'
              f'\033[35mP-Val:\033[0m {p_val:.4f}\n'
              f'\033[35mP-Threshold:\033[0m {p_threshold:.4f}')
    
    # Return statistical test results
    return stat_results


def analyze_kruskal_wallis(pdDataFrame:pd.DataFrame,
                           feature:str,
                           target:str='attack',
                           p_threshold:float=0.05) -> dict:
    '''
    About
    -----
    - Applies scipy's kruskal wallis on a numerical feature and prints a color-coded statistic and p_value
      where green is the p_value is less than p_threshold, red otherwise
    - Returns a dictionary of all the information derived from the kruskal wallis function
    - This is intended to statistically analyze NUMERICAL features

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Pandas dataframe to apply the statistic test on

    - feature (str) :
        - The name of the feature (column) to be tested with "target"
    
    - target (str) :
        - Default: attack
        - The name of the target feature (column) to be tested with "feature"

    - p_threshold (float) :
        - Default: 0.05
        - The value to determine if the statistic is statistically significant or not (p_val <= p_threshold means statistically significant)

    Returns
    -------
    - A color-coded statistic and p_value where green means p_val <= p_threshold using scipy's kruskal wallis as a print statement
    - A dictionary holding statistical test information
    '''
    # Setup and conduct test
    groups = [group[feature].values for name, group in pdDataFrame.groupby(target)]
    stat, p_val = kruskal(*groups)
    
    # Create dictionary of outputs
    stat_results = {
        'feature': feature,
        'target': target,
        'stat': stat,
        'p_val': p_val
    }

    # Print off color-coded results
    if p_val <= p_threshold:
        print(f'\033[32m{feature} IS statistically significant with {target}!\033[0m\n'
              f'\033[35mStatistic:\033[0m {stat:.4f}\n'
              f'\033[35mP-Val:\033[0m {p_val:.4f}\n'
              f'\033[35mP-Threshold:\033[0m {p_threshold:.4f}\n')
    else:
        print(f'\033[33m{feature} IS NOT statistically significant with {target}!\033[0m\n'
              f'\033[35mStatistic:\033[0m {stat:.4f}\n'
              f'\033[35mP-Val:\033[0m {p_val:.4f}\n'
              f'\033[35mP-Threshold:\033[0m {p_threshold:.4f}\n')
    
    # Return statistical test results
    return stat_results

# =================================================================================================
# END Statistical Analysis Functions
# START Associativity Functions
# =================================================================================================

def analyze_cramers(pdDataFrame:pd.DataFrame,
                    feature:str,
                    target:str='attack',
                    association_groups:list[float]=[0.5, 0.3, 0.1],
                    analyze_chi_square_dict:dict=None) -> float:
    '''
    About
    -----
    - Applies the cramer's formula to determine the associativity of the categorical "feature" with "target" and prints a color-coded association and returns the result
    - "association_groups" correlate with the color-coding of the print statement (green>=0.5, cyan>=0.3, yellow>=0.1, red>=0.0)
    - This is intended to statistically analyze the associativity of a CATEGORICAL feature

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Pandas dataframe to apply the statistic test on

    - feature (str) :
        - The name of the feature (column) to be tested with "target"
    
    - target (str) :
        - Default: attack
        - The name of the target feature (column) to be tested with "feature"

    - association_groups (list[float]) :
        - Default: [0.5, 0.3, 0.1]
        - WILL EXIT FUNCTION IF len(association_gropus) != 3
        - This is generally to flag high/medium/low associativity
        - Determines the color-coding of the print statement according to the produced result
        - Please be aware that this assumes the level of associativity from highest to lowest, so please ensure the list is following the same pattern!
        - GREEN>=0.5, CYAN>=0.3, YELLOW>=0.1, RED>=0.0

    - anaylze_chi_square_dict (dict) :
        - Default: None
        - This is intended to be the dictionary output from the "analyze_chi_square" function to expedite intial variable derivations
        - If None, it just obtains the contingency table and runs scipy's "chi2_contingency" function

    Raises
    ------
    - ValueError :
        - If len(association_groups) != 3

    Returns
    -------
    - A color-coded statistic where (green>=0.5, cyan>=0.3, yellow>=0.1, red>=0.0)
    - The produced result (float) from application of cramer's formula
    '''
    # ----- Check Valid association_groups --------------------------------------------------------
    if len(association_groups) != 3:
        raise ValueError('\033[31mLength of association_groups is NOT equal to 3!\n'
                         'Cancelling execution of analyze_cramers function!\033[0m')
    
    # ----- Obtain Initial Variables For Test -----------------------------------------------------
    if analyze_chi_square_dict:
        stat = analyze_chi_square_dict['stat']
        n = analyze_chi_square_dict['contingency_table'].sum().sum()
        phi2 = stat / n
        r, k = analyze_chi_square_dict['contingency_table'].shape
    else:
        contingency_table = pd.crosstab(pdDataFrame[feature], pdDataFrame[target])
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
    
    # ----- Conduct Test --------------------------------------------------------------------------
    # Correct for bias
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    # produce result
    result = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    # ----- Print/Return Results ------------------------------------------------------------------
    # Green statement (High associativity)
    if result >= association_groups[0]:
        print(f'\033[32m{feature} has HIGH ASSOCIATION with {target}!')

    # Cyan statement (Medium associativity)
    elif result >= association_groups[1]:
        print(f'\033[36m{feature} is MODERATE ASSOCIATION with {target}!')

    # Yellow statement (Low associativity)
    elif result >= association_groups[2]:
        print(f'\033[33m{feature} has LOW ASSOCATION with {target}!')

    # Red statement (No associativity)
    else:
        print(f'\033[31m{feature} has MINIMAL ASSOCATION with {target}!')

    # Return result
    return result


def analyze_mutual_information(pdDataFrame:pd.DataFrame,
                               feature:str,
                               target:str='attack',
                               association_groups:list[float]=[0.5, 0.3, 0.1],
                               n_samples=100000) -> dict:
    '''
    About
    -----
    - Applies the sklearn's mutual information function to determine the associativity of the numerical "feature" with "target" and prints a color-coded association and returns the result
    - "association_groups" correlate with the color-coding of the print statement (green>=0.5, cyan>=0.3, yellow>=0.1, red>=0.0)
    - This is intended to statistically analyze the associativity of a NUMERICAL feature

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Pandas dataframe to apply the statistic test on

    - feature (str) :
        - The name of the feature (column) to be tested with "target"
    
    - target (str) :
        - Default: attack
        - The name of the target feature (column) to be tested with "feature"

    - association_groups (list[float]) :
        - Default: [0.5, 0.3, 0.1]
        - WILL EXIT FUNCTION IF len(association_gropus) != 3
        - This is generally to flag high/medium/low associativity
        - Determines the color-coding of the print statement according to the produced result
        - Please be aware that this assumes the level of associativity from highest to lowest, so please ensure the list is following the same pattern!
        - GREEN>=0.5, CYAN>=0.3, YELLOW>=0.1, RED>=0.0

    - n_samples (int) :
        - Default: 100000
        - The number of samples to stratify over the "target" feature for testing

    Raises
    ------
    - ValueError :
        - If len(association_groups) != 3

    Returns
    -------
    - A color-coded statistic where (green>=0.5, cyan>=0.3, yellow>=0.1, red>=0.0)
    - The produced result (dict) from application of sklearn's mutual information function
    '''
    # ----- Check Valid association_groups --------------------------------------------------------
    if len(association_groups) != 3:
        raise ValueError('\033[31mLength of association_groups is NOT equal to 3!\n'
                         'Cancelling execution of analyze_cramers function!\033[0m')
    
    # ----- Setup and Perform MI Test -------------------------------------------------------------
    y_all = pdDataFrame[target]
    
    # Prepare X (feature) and y (target)
    X_sample, _, y_sample, _ = train_test_split(pdDataFrame[[feature]], 
                                                y_all, 
                                                train_size=n_samples, 
                                                stratify=y_all, 
                                                random_state=42)
    
    # Encode "target"
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_sample)
    
    # Compute mutual information
    mi_score = mutual_info_classif(X_sample, y_encoded, discrete_features=[False], random_state=42)
    
    # ----- Print and Return Results --------------------------------------------------------------
    # Create result dictionary
    results = {
        'feature': feature,
        'target': target,
        'mi-score': mi_score[0]
    }

    # Green statement (High associativity)
    if mi_score[0] >= association_groups[0]:
        print(f'\033[32m{feature} has HIGH ASSOCIATION with {target}!')

    # Cyan statement (Medium associativity)
    elif mi_score[0] >= association_groups[1]:
        print(f'\033[36m{feature} is MODERATE ASSOCIATION with {target}!')

    # Yellow statement (Low associativity)
    elif mi_score[0] >= association_groups[2]:
        print(f'\033[33m{feature} has LOW ASSOCATION with {target}!')

    # Red statement (No associativity)
    else:
        print(f'\033[31m{feature} has MINIMAL ASSOCATION with {target}!')

    # Return result
    return results

# =================================================================================================
# END Associativity Functions
# START Dual-Purpose Function
# =================================================================================================

def analyze_statistical_significance_and_associativity(pdDataFrame:pd.DataFrame,
                                                       feature:str,
                                                       target:str='attack',
                                                       is_object_feature:bool=False):
    '''
    About
    -----
    - Runs through a categorical or numerical statistical and associative analysis along with visuals
    - Uses the following functions:
        - analyze_chi_square
        - vis_categorical
        - analyze_cramers
        - vis_singular_association
        - analyze_kruskal_wallis
        - vis_numerical
        - analyze_mutual_information

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to perform the analysis and visuals on

    - feature (str) :
        - The feature (column) name to be analyzed with "target"

    - target (str) :
        - The target feature name to be analyzed with "feature

    - is_object_feature (bool) :
        - Default: False
        - Whether or not the feature to analyze is meant to be analyzed as a categorical (object/str) or numerical (int/float) way

    Returns
    -------
    - A complete statistical and associative analysis along with visuals for a single categorical/numerical feature
    '''
    # Run categorical-based analysis
    if is_object_feature:
        results = analyze_chi_square(pdDataFrame, feature, target)
        vis_categorical(pdDataFrame, feature, target)
        score = analyze_cramers(pdDataFrame, feature, target, analyze_chi_square_dict=results)
        vis_singular_association(feature, score, 'Cramers')

    # Run numerical-based analysis
    else:
        results = analyze_kruskal_wallis(pdDataFrame, feature, target)
        vis_numerical(pdDataFrame, feature, target)
        score = analyze_mutual_information(pdDataFrame, feature, target)
        vis_singular_association(feature, score['mi-score'])

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
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to base the visualization off of

    - col_for_comparison (str) :
        - The column name intended to use for comparison to the "target" column

    - target (str) :
        - Default: attack
        - The name of the target column name for comparison

    Returns
    -------
    - Violin plot of the top 2, bottom 2, and middle most distibution of the "target" tags with the "col_for_comparison"
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

    # Prevent unreasonably large numbers that will cause errors
    plot_df[col_for_comparison] = plot_df[col_for_comparison].clip(upper=1e12)
    
    # Sanity check for nans and infs
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_for_comparison])

    # Interpret zeros for log scale (Crashes otherwise)
    # We add 1 to all values so that 0 becomes log(1) = 0. 
    if (plot_df[col_for_comparison] <= 0).any():
        plot_df[col_for_comparison] = plot_df[col_for_comparison] + 1.0
        title_suffix = " (Log Scale: x+1 offset applied)"
    else:
        title_suffix = ""

    # Handle extremely close to 0 float values
    plot_df = plot_df[np.isfinite(plot_df[col_for_comparison])]
    plot_df = plot_df[plot_df[col_for_comparison] > 1e-30]

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
    plt.title(f'Numerical Signature: {col_for_comparison} vs {target}')
    plt.tight_layout()
    plt.show()


def vis_categorical(pdDataFrame:pd.DataFrame, col_for_comparison: str, target: str = 'attack') -> None:
    '''
    About
    -----
    - Visualizes a heatmap of categorical data where the top 2, bottom 2, and middle most distributed attack types are displayed
      along with the top 2, bottom 2, and middle col_for_comparison values

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - The Pandas dataframe to base the visualization off of

    - col_for_comparison (str) :
        - The column name intended to use for comparison to the "target" column

    - target (str) :
        - Default: attack
        - The name of the target column name for comparison

    Returns
    -------
    - Heatmap of the top 2, bottom 2, and middle most distibution of the "target" tags with the "col_for_comparison" with the top 2, bottom 2, and middle most values
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

    # Determine distribution of col_for_comparison
    counts = plot_df[col_for_comparison].value_counts()
    total = len(plot_df)
    percentages = (counts / total * 100).round(2).to_dict()

    # Define the top, middle, and most rare values
    top_2_val = counts.index[:2]
    middle_val = [counts.index[len(counts)//2]]
    bottom_2_val = counts.index[-2:]

    # Create a sub_dataframe to expedite lookup
    target_list = list(top_2_val) + list(middle_val) + list(bottom_2_val)
    plot_df = plot_df[plot_df[col_for_comparison].isin(target_list)].copy()

    # Create value label and distribution percentage
    plot_df[f'{col_for_comparison} Value'] = plot_df[col_for_comparison].apply(lambda x: f"{x} ({percentages[x]}%)")

    # Acquire the crosstab
    ct = pd.crosstab(plot_df['Attack Type'], plot_df[f'{col_for_comparison} Value'], normalize='index')

    # Plot the heatmap
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        ct, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Probability Density'}
    )
    
    plt.title(f'Categorical Signature: {col_for_comparison} vs {target}')
    plt.ylabel('Attack Type (Global Distribution %)')
    plt.xlabel(f'{col_for_comparison} (Global Distribution %)')
    plt.tight_layout()
    plt.show()


def vis_singular_association(feature_name:str,
                             score:float,
                             metric_type:str = "Mutual Information") -> None:
    '''
    About
    -----
    - Visualizes the association score of a single feature against the target using a bar graph
    - Categorizes the strength into High, Medium, Low, or Minimal tiers

    Parameters
    ----------
    - feature_name (str) :
        - Name of the feature being tested for association to some target feature

    - score (float) :
        - Score of either the analyze_cramers or analyze_mutual_information functions

    - metric_type (str) :
        - The type of evaluation used (Ideally Cramers or Mutual Information)

    Returns
    -------
    - Bar graph visualization of the associativity of "feature_name"
    '''
    # ----- Determine Association Color -----------------------------------------------------------
    if score >= 0.5:
        tier, color = "High", "#2ecc71"
    elif score >= 0.3:
        tier, color = "Moderate", "#00FFFF" 
    elif score >= 0.1:
        tier, color = "Low", "#f1c40f"
    else:
        tier, color = "Minimal", "#e74c3c"

    # ----- Setup and Show Visualization ----------------------------------------------------------
    # Visualization size
    plt.figure(figsize=(8, 3))
    
    # Background bar
    plt.barh([feature_name], [1.0], color='#ecf0f1', label='Scale Range')

    # Score bar
    plt.barh([feature_name], [score], color=color, label=f'{tier} Associativity')

    # Annotations and visual-aids
    plt.text(score + 0.02, 0, f"{score:.4f} ({tier})", va='center', fontweight='bold')
    plt.axvline(x=0.1, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0.3, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)

    # Create labels and ranges
    plt.title(f"{metric_type} Analysis: {feature_name}", loc='left', fontsize=12)
    plt.xlim(0, 1.0)
    plt.xlabel("Association Score (0.0 - 1.0)")
    plt.legend(loc='lower right')

    # Show visualization
    plt.tight_layout()
    plt.show()

# =================================================================================================
# START Visualization Functions
# END Preprocess Functions
# =================================================================================================

def preprocess_normal_data(pdDataFrame:pd.DataFrame) -> pd.DataFrame:
    '''
    About
    -----
    - Reduces, encodes, and scales the normal dataframe based off of analysis in "explore_normal_data.ipynb"

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame) :
        - Original Pandas dataframe to reduce further in preparation for model training

    Returns
    -------
    - reduced_df (pd.DataFrame) :
        - Reduced, encoded, and scaled Pandas dataframe based off of analysis in "explore_normal_data.ipynb"
    '''
    # Reduce df to identified columns to keep
    cols_to_keep = [
        'in_bytes',
        'out_bytes',
        'flow_duration_milliseconds',
        'duration_in',
        'min_ttl',
        'max_ttl',
        'longest_flow_pkt',
        'shortest_flow_pkt',
        'max_ip_pkt_len',
        'src_to_dst_avg_throughput',
        'num_pkts_up_to_128_bytes',
        'num_pkts_128_to_256_bytes',
        'tcp_win_max_in',
        'attack'
    ]
    reduced_df = pdDataFrame[cols_to_keep]

    # Encode the Target
    le = LabelEncoder()
    reduced_df['target'] = le.fit_transform(reduced_df['attack'])
    
    # Apply log transform large value cols first
    log_cols = ['in_bytes', 'out_bytes', 'src_to_dst_avg_throughput', 'flow_duration_milliseconds']
    for col in log_cols:
        # np.log1p handles the log(x+1) to avoid -inf on zero values
        reduced_df[col] = np.log1p(reduced_df[col].clip(upper=1e12)) 

    # Apply Robust Scaling to everything else
    scaler = RobustScaler()
    numerical_features = reduced_df.select_dtypes(include=[np.number]).columns.drop('target')
    reduced_df[numerical_features] = scaler.fit_transform(reduced_df[numerical_features])
    
    return reduced_df

# =================================================================================================
# START Preprocess Functions
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