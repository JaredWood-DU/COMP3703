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
- [x] Clean Dataset
- [x] Exploration of Original Dataset
- [x] Prepare Dataset
	- [x] Apply Encoding, normalization, feature engineering where necessary from above exploration
	- [x] Implement Complex Network Features
		- [x] Determine if iGraph is faster than networkX
		- [x] Determine feasability of an ideal datasize for feature engineering
		- [x] Determine if graph is directed or undirected
		- [x] Slim-down dataset initially if necessary
		- [x] Determine 0.0.0.0 use-case
		- [x] Generate baseline information
		- [x] Engineer 1st and 2nd Eigenvalues (Defines underlying network architecture.  Theoretically, establishes a sort of baseline normalcy in activity)
		- [x] Engineer 1st and 2nd Eigenvectors (Theoretically should denote inconsistent entries with other nodes, basically if this node potentially is malicious.  In other words, should denote a unique footprint for every entry that can be compared to the baseline normalcy)
		- [x] Explore feasability of PageRank (Random Walks for convergence detection.  Theoretically, benign should be clustered together and the more "stealthy" an attack is, the further they are from this cluster.  DDoS likely will be the closest furthest away attack and BotNets will likely be the furthest isolated point from this cluster)
		- [x] Encode and normalize information where necessary (Large values like bytes may need to be normalized as the eigenvalues will likely be only single digits)
- [x] Select Models
- [x] Train and Test Models
	- [x] Baseline Model (Vanilla training of model)
	- [x] Best Tabular-Centric Model (No Complex Network Features)
	- [x] Best Network Model (Only Complex Network Features)
	- [ ] Best Hybrid Model (Both Tabular and Network Features)
- [x] Observe Key Differences
- [ ] Create Visualizations
- [ ] Create Final Notebook Presentation





## Known Issues

[Back to Table of Contents](#table-of-contents)

- 

