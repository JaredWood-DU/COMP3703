# COMP3703 - Homework





## Table of Contents

- [Environment Setup](#environment-setup)
- [Known Issues](#known-issues)
  - [HW2 Issues](#hw2-issues)





## Environment Setup

[Back to Table of Contents](#table-of-contents)

You have two options for setting up your Python environment:

### Option 1: Conda (Recommended)

**Conda** is an open-source environment and package manager that makes it easy to manage Python versions and dependencies. If you do not already use an environment manager, you may want to familiarize yourself with one since it helps avoid conflicts and makes reproducibility easier.  I use Conda and I think it's the easiest (Though I haven't used other packages)

**Steps:**
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository (Or just download ```environment.yml```).
3. Navigate to the `homework` directory.
4. Create the environment using the provided `environment.yml`:
	```bash
	conda env create -f environment.yml
	conda activate COMP3703
	```

### Option 2: pip (Use with Caution)

You can also use `pip` with the `requirements.txt` file. **Note:** Your current Python version may be incompatible with some packages (e.g., `temporian` requires Python 3.8-3.11). Using pip does not manage Python versions, so you must ensure your Python version matches the requirements.

**Steps:**
1. Ensure you are using a compatible Python version (see above).
2. Clone this repository (Or just download requirements.txt).
3. Navigate to the `homework` directory.
4. Install dependencies:
	```bash
	pip install -r requirements.txt
	```






## Known Issues

[Back to Table of Contents](#table-of-contents)

These generally aren't really issues as they are additional imports and aliases to ensure the file runs that were absent originally

[HW 2 Issues](#hw2-issues)

### HW2 Issues

[Back to Known Issues](#known-issues)

- Imports and import aliases missing (Added at top of file)
- If installing ```temporian```, your python version may be incompatable (```python==3.11.13``` works, this is also the base for the environment)

