# Opinion Network Formation - Assignment 1

This repository contains the data processing pipeline, network construction, and metric analysis for the class Opinion Network assignment. 

## Local Setup

To run the analysis notebooks, you must set up a local Python virtual environment to manage dependencies.

### 1. Initialize the Virtual Environment
Run the following command in the root directory of this project to create an isolated environment:
```bash
python3 -m venv .venv

```

### 2. Activate the Environment

You must activate the environment before installing packages or running the code.

* **macOS/Linux:**
```bash
source .venv/bin/activate

```


* **Windows:**
```bash
.venv\Scripts\activate

```



### 3. Install Required Modules

Once the virtual environment is active (your terminal prompt will usually show `(.venv)`), install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn networkx matplotlib ipykernel

```

## Module Overview

* **`pandas`**: Used for loading the raw `Survey_Results_UC.csv` dataset, handling missing values, and mapping ordinal Likert-scale strings to numerical integers.
* **`numpy`**: Provides support for efficient matrix operations and handling the adjacency matrix diagonals.
* **`scikit-learn`**: Utilized specifically for `sklearn.metrics.pairwise.cosine_similarity` to calculate the mathematical distance between respondent vectors.
* **`networkx`**: The core graph library used to construct the undirected network, filter isolated nodes, detect Louvain communities, and extract global/local metrics (density, centrality, modularity).
* **`matplotlib`**: Renders the 2D network visualizations using force-directed layouts (e.g., `spring_layout`).

