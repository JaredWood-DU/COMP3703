"""
FILE OVERVIEW:
- Underlying code intended for use in wrangle.ipynb to keep notebook cleaner
- Consists of functions that acquire the intended dataset and prepares the dataset by handling missing/problematic entries
- DATASET: https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset?resource=download

=================================================

MISC COMMENTS:
- This is a very large dataset, please be aware that depending on your resources, doing the entire 'wrangle' process may take a few hours

=================================================

FILE CONTENTS:
- File Overview, Imports, Global Variables
- Acquire Dataset Functions
    - download_raw
    - create_raw_short
- Prepare Dataset Functions
    - diag_missing_values
    - handle_missing_values
- Main Function
"""
# ----- Imports -----------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

# ----- Global Variables --------------------------------------------------------------------------
# NA

# =================================================================================================
# END File Overview, Imports, Global Variables
# START Acquire Dataset Functions
# =================================================================================================

def download_raw(filename:str='raw_full.csv') -> None:
    """
    About
    -----
    - If the [filename] does not exist locally, then download the raw version of the 
    'NF-UQ-NIDS-v2 Network Intrusion Detection Dataset' and moves it to this local directory
    - URL: https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset?resource=download

    Parameters
    ----------
    - filename (str): 
        - Default: raw_full.csv
        - The name of the raw dataset file

    Returns
    -------
    - Downloads the raw datafile and moves it to this local directory if it doesn't already exist
    """
    # ----- Check If filename Already Exists ------------------------------------------------------
    if os.path.exists(filename):
        print('\033[32m'
              f'{filename} exists already!  No need to download the datafile again!'
              '\033[0m')
        return
    
    # ----- Try Downloading Dataset ---------------------------------------------------------------
    try:
        # Notify user of download attempt
        print('\033[33m'
              f'{filename} not found, attempting to download the raw dataset...'
              '\033[0m')

        # Download latest version
        download_dir = kagglehub.dataset_download("aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset")

        # Move the downloaded file to this working directory
        download_filename = os.listdir(download_dir)[0]
        download_direct_path = f'{download_dir}/{download_filename}'
        new_direct_path = f'{os.getcwd()}/{filename}'
        os.rename(download_direct_path, new_direct_path)

        # Notify user of pathing of the moved file and completed process
        print("Path to dataset files:", new_direct_path)
        print('\033[32m'
              f'Successfully downloaded the raw datafile as {filename}!'
              '\033[0m')
        return

    # ----- Notify Failure To Download Dataset ----------------------------------------------------
    except Exception as e:
        print('\033[31m'
              'Failed to download the raw dataset!\n'
              'Exception:'
              '\033[0m'
              f'{e}')


def create_raw_short(
        full_raw_filename:str='raw_full.csv',
        filename:str='raw_short.csv',
        reduction_percent:float=0.10,
        target_variable:str='Attack') -> None:
    '''
    About
    -----
    - Creates a shorter version of [full_raw_filename] as [filename] that is [reduction_percent] of [full_raw_filename]
    - Will check to see if [filename] exists first, if not, calls download_raw()
    - Uses pandas dataframe to stratify across the target variable, in this case, 

    Parameters
    ----------
    - full_raw_filename (str): 
        - Default: raw_full.csv
        - The name of complete dataset in the current working directory to be reduced
    - filename (str):
        - Default: raw_short.csv
        - The name of the reduced dataset in the current working directory
    - reduction_percent (float): 
        - Default: 0.10
        - The percentage of the [full_raw_filename] to be transferred to [filename]
    - target_varialbe (str): 
        - Default: Attack
        - The target variable (Column/Feature name) to stratify over

    Raises
    ------
    - KeyError
        - If [target_variable] cannot be found in the column names of [full_raw_filename]

    Returns
    -------
    - A reduced version of [full_raw_filename] as [filename] to help expedite exploration and coding
    '''
    # ----- Check If filename Already Exists ------------------------------------------------------
    if os.path.exists(filename):
        print('\033[32m'
              f'{filename} exists already!  No need to re-create the reduced datafile again!'
              '\033[0m')
        return
        
    # ----- Check If full_raw_filename Already Exists ---------------------------------------------
    if not os.path.exists(full_raw_filename):
        print('\033[33m'
              f'{full_raw_filename} does not exist, downloading it first in order to create reduced version...'
              '\033[0m')
        
        try:
            download_raw(filename=full_raw_filename)

        except Exception as e:
            return

    # ----- Try Creating Reduced Dataset ----------------------------------------------------------
    try:
        print('\033[33m'
              f'{full_raw_filename} exists, attempting to create reduced version as {filename}...'
              '\033[0m')
        raw_df = pd.read_csv(full_raw_filename)

        # Check if the target_variable exists first
        if target_variable not in raw_df.columns:
            raise KeyError('\033[31m'
                           f'{target_variable} does not exist in the column names of {full_raw_filename}!'
                           f'Detected Column Names: {list(raw_df.columns)}'
                           '\033[0m')
        
        # Split raw_df into the reduced_df using sklearn's function
        _, reduced_df = train_test_split(
            raw_df,
            test_size=reduction_percent,
            stratify=raw_df[target_variable],
            random_state=3703
        )

        # Export the reduced dataset as [filename]
        pd.DataFrame.to_csv(reduced_df, path_or_buf=filename)
        print('\033[32m'
              f'Successfully reduced {full_raw_filename} into {filename} at {reduction_percent*100}% the original size and stratified over {target_variable}!'
              '\033[0m')
        return

    # ----- Notify Failure To Create Reduced Dataset ----------------------------------------------
    except Exception as e:
        print('\033[31m'
              f'Failed to create reduced dataset!\n'
              'Exception:'
              '\033[0m'
              f'{e}')

# =================================================================================================
# END Acquire Dataset Functions
# START Prepare Dataset Functions
# =================================================================================================

def diag_missing_values(pdDataFrame:pd.DataFrame=None, dataFileName:str='raw_short.csv') -> pd.DataFrame:
    '''
    About
    -----
    - Reads in [pdDataFrame] and performs a diagnosis of any and all hard/soft null values, duplicates, and returns this information
    - If [pdDataFrame] is None, this will attempt to create the Pandas DataFrame with [dataFileName]
    - If [pdDataFrame] is None AND [dataFileName] is not found, this function will cease further operations
    - This will return the used Pandas DataFrame
    - Hard Nulls (Generally resolved with pd.DataFrame.fillna()):
        - pdDataFrame.isna() (This is the generic method to catch things like NaN, None, NaT)
    - Soft Nulls (Requires further examination to determine resolution):
        - Numeric Checks:
            - inf (Infinte values)
        - IP Checks:
            - 0.0.0.0 (Invalid IP)
        - Non-Negative Col Check:
            - -1 (Not Applicable)
        - String Checks:
            - "" (Empty strings)
            - " " (Whitespace strings)
            - "?" (Kaggle specific for missing data)
            - "None" (Sanity Check)
            - "nan" (Sanity Check)
            - "NULL" (Sanity Check)

    Parameters
    ----------
    - pdDataFrame (Pandas DataFrame):
        - Default: None
        - If not None, will use this to perform the diagnosis and produce results
        - If None, [dataFileName] will be used to create the Pandas DataFrame
    - dataFileName (str):
        - Default: raw_short.csv
        - The filename to search for locally to create [pdDataFrame] if necessary

    Returns
    -------
    - PRINT STATEMENT
        - Diagnosis of missing/problematic entries found in [pdDataFrame]
    - pd.Dataframe:
        - Whatever Pandas DataFrame that was used during the diagnosis
    '''
    # ----- Check If pdDataFrame Is Given ---------------------------------------------------------
    if not pdDataFrame:
        print('\033[33m'
              f'pdDataFrame not provided, attempting to create Pandas DataFrame using {dataFileName}...'
              '\033[0m')
        
        try:
            if not os.path.exists(dataFileName):
                raise FileNotFoundError('\033[31m'
                                        f'Unable to find {dataFileName}, cannot establish pdDataFrame!\n'
                                        f'Current working directory: {os.getcwd()}'
                                        '\033[0m')
            pdDataFrame = pd.read_csv(dataFileName)
        
        except Exception as e:
            print('\033[31m'
                  f'Failed to establish pdDataFrame!'
                  'Exception:'
                  '\033[0m'
                  f'{e}')

    # ----- Initialize Diagnosis ------------------------------------------------------------------
    diagDict = {}
    duplicates = pdDataFrame.duplicated().sum()
    
    # Soft Null Checklist
    stringChecks = ['', ' ', '?', 'None', 'nan', 'NULL']
    ipPlaceholder = '0.0.0.0'
    negPlaceholder = -1

    # ----- Perform Diagnosis Loop ----------------------------------------------------------------
    for col in pdDataFrame.columns:
        # 1. Hard Nulls (Standard NaN/None)
        hardNulls = pdDataFrame[col].isna().sum()
        
        # 2. Soft Nulls - Initialize counts
        infCount = 0
        ipCount = 0
        negCount = 0
        strCount = 0

        # Numeric Checks (Infinities and Negative placeholders)
        if pd.api.types.is_numeric_dtype(pdDataFrame[col]):
            infCount = np.isinf(pdDataFrame[col]).sum()
            negCount = (pdDataFrame[col] == negPlaceholder).sum()
        
        # String/Object Checks
        else:
            strCount = pdDataFrame[col].isin(stringChecks).sum()
            # Specific IP Check
            if 'ADDR' in col.upper() or 'IP' in col.upper():
                ipCount = (pdDataFrame[col] == ipPlaceholder).sum()

        diagDict[col] = {
            'hardNulls': hardNulls,
            'infs': infCount,
            'negatives': negCount,
            'placeholders': strCount,
            'bad_ips': ipCount
        }

    # ----- Print Diagnosis Information -----------------------------------------------------------
    # Print overview of what is being diagnosed
    print('\033[35m'
          'What Is Being Diagnosed:\n'
          '\tDuplicate rows check\n'
          '\tHard Nulls:\t pd.isna()\n'
          '\tNumeric Checks:\t np.isinf()\n'
          f'\tIP Checks:\t{ipPlaceholder}\n'
          f'\tNonsense Vals:\t{negPlaceholder}\n'
          f'\tBad Strings:\t{stringChecks}'
          '\033[0m')
    
    # Print out duplicates
    if duplicates == 0:
        print('\033[32m'
              'No duplicates found!'
              '\033[0m')
        
    # Print off all information
    for col in diagDict:
        totalNulls = sum([diagDict[col][errorCat] for errorCat in diagDict[col]])
        if totalNulls == 0:
            print('\033[32m'
                  f'{col} had 0 detected nulls!'
                  '\033[0m')
        else:
            print('\033[33m'
                f'{col} detected:'
                '\033[0m')
            for errorCat in diagDict[col]:
                if diagDict[col][errorCat] == 0:
                    continue
                else:
                    print(f'\t{errorCat}: {diagDict[col][errorCat]}')

    return pdDataFrame


def handle_missing_values(pdDataFrame:pd.DataFrame, filename:str='prepared_short.csv') -> None:
    '''
    About
    -----
    - From exploration of diag_missing_values(), this specifically handles what was detected
    - From diag_missing_values():
        - No duplicates
        - IPV4_SRC_ADDR: 0.0.0.0 detected
        - IPV4_DST_ADDR: 0.0.0.0 detected
    - Nothing will be adjusted from the raw files since there could be a correlation with bad IPs and malicious activity

    Parameters
    ----------
    - pdDataFrame (pd.DataFrame):
        - The Pandas DataFrame to prepare accordingly and save as [filename] to load in prepared datasets in the future
    - filename (str):
        - Default: prepared_short.csv
        - The name of the prepared dataset to be loaded later

    Returns
    -------
    - DATA FILES
        - [filename] is the prepared dataset saved locally
    '''
    # ----- Check If filename Already Exists ------------------------------------------------------
    if os.path.exists(filename):
        print('\033[32m'
              f'{filename} already exists!  No need to re-create it!'
              '\033[0m')
        return

    # ----- Create Prepared filename --------------------------------------------------------------
    try:
        print('\033[33m'
            f'Attempting to create the prepared dataset as {filename}...'
            '\033[0m')
        pd.DataFrame.to_csv(pdDataFrame, path_or_buf=filename)
        print('\033[32m'
            f'Successfully created the prepared dataset {filename}'
            '\033[0m')
        
    except Exception as e:
        print('\033[31m'
              'Failed to create prepared dataset!\n'
              'Exception:'
              '\033[0m'
              f'{e}')

# =================================================================================================
# END Prepare Dataset Functions
# START Main Function
# =================================================================================================

def main():
    '''
    About
    -----
    - Simply just runs the entire wrangle process from start to finish with default values
    - Acquire:
        - download_raw()
        - create_raw_short()
    - Prepare:
        - diag_missing_values()
        - handle_missing_values()

    Returns
    -------
    - Everything from start to finish in the wrangle process
    '''
    download_raw()
    create_raw_short()
    df = diag_missing_values()
    handle_missing_values(df)

# =================================================================================================
# END Main Function
# =================================================================================================