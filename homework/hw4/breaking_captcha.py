"""
FILE OVERVIEW:
- 

=================================================

MISC COMMENTS:
- 

=================================================

FILE CONTENTS:
1. File Overview, Imports, Global Variables
2. Helper Functions
    - Thing 1
    - Thing 2
3. Main Function
"""
# ----- Imports -----------------------------------------------------------------------------------


# ----- Global Variables --------------------------------------------------------------------------


# =================================================================================================
# END File Overview, Imports, Global Variables
# START Helper Functions
# =================================================================================================

def thing1():
    """
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
    """
    pass


def thing2():
    pass

# =================================================================================================
# END Helper Functions
# START Main Function
# =================================================================================================

def main():
    """
    - This wrapper function is expecting the 'ray_nn_train_func' to return a dictionary of what info to log,
      the structure of the return dictionary should look similar to config.yaml['nn_ray_train_config']['metrics']

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
    """
    pass

# =================================================================================================
# END Main Function
# =================================================================================================