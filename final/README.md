# Intrusion Detection Systems with AI and Complex Networks

The intent is primarily research-based and information-gain.  

This project explores a large dataset of cybersecurity information and observes various standard AI methodology (e.g. optimization, model selection, hyperparameterization, etc.) and then uses mathematic complex network concepts to feature engineer additional information in order to analyze differences in performances on classification of various types of cybersecurity attacks, to include benign attacks.  

With this, the idea is to observe a baseline model, best-performing AI model WITHOUT complex network information, and best-performing AI model WITH complex network information.  

Hypothetically, dense information engineered from complex networks should result in a better performing model inherently because the nature of how AI "learns" is mere pattern recognition and brute forcing statistical weights to arrive at the desired outcome.  Thus weakly-connected information will lead to weakly-connected patterns (In which the AI will force a pattern where one shouldn't exist) and strongly-connected information (e.g. spectral graph theory) should lead to strongly-connected patterns



## Table of Contents

- [Environment Setup](#environment-setup)
- [Objectives](#objectives)
- [Known Issues](#known-issues)





## Environment Setup

[Back to Table of Contents](#table-of-contents)

You have two options for setting up your Python environment:

### Option 1: Conda (Recommended)

**Conda** is an open-source environment and package manager that makes it easy to manage Python versions and dependencies. If you do not already use an environment manager, you may want to familiarize yourself with one since it helps avoid conflicts and makes reproducibility easier.  I use Conda and I think it's the easiest (Though I haven't used other packages)

**Steps:**
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository (Or just download ```environment.yml```).
3. Navigate to the `final` directory.
4. Create the environment using the provided `environment.yml`:
	```bash
	conda env create -f environment.yml
	conda activate COMP3703
	```

### Option 2: pip (Use with Caution)

You can also use `pip` with the `requirements.txt` file. Using pip does not manage Python versions, so you must ensure your Python version matches the requirements.

**Steps:**
1. Ensure you are using a compatible Python version (see above).
2. Clone this repository (Or just download requirements.txt).
3. Navigate to the `final` directory.
4. Install dependencies:
	```bash
	pip install -r requirements.txt
	```






## Objectives

[Back to Table of Contents](#table-of-contents)

- [x] Acquire Dataset
- [ ] Prepare Dataset
- [ ] Exploration of Original Dataset
- [ ] Adjust Dataset If Necessary
- [ ] Implement Complex Network Features
- [ ] Select Models
- [ ] Train and Test Models
	- [ ] Baseline Model (Vanilla training of model)
	- [ ] Best Tabular-Centric Model (No Complex Network Features)
	- [ ] Best Network Model (Only Complex Network Features)
	- [ ] Best Hybrid Model (Both Tabular and Network Features)
- [ ] Observe Key Differences
- [ ] Create Visualizations
- [ ] Create Final Notebook Presentation





## Known Issues

[Back to Table of Contents](#table-of-contents)

- Re-prepare dataset to remove duplicate rows and normalize feature names to be lowercase (Because I don't want to press the shift key)

