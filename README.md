# Opinion Network Formation - Assignment 1

This repository contains the data processing pipeline, network construction, and metric analysis for the class Opinion Network assignment. 

## Local Setup

To run the analysis notebook and scripts, you must set up a local Python virtual environment to manage dependencies.

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

Once the virtual environment is active (your terminal prompt will usually show `(.venv)`), install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

```

Installing from `requirements.txt` rather than naming packages by hand keeps everyone on the same versions, and pins `ipykernel` below version 7 — the 7.x releases cannot be started by the VS Code Jupyter extension.

### 4. Run the Analysis

With the environment active, run the network analysis script from the project root:

```bash
python network_analysis.py

```

It prints every metric to the terminal and writes the five figures to `figures/`. The narrative interpretation of the results is in `RESULTS.md`.

If you get a `ModuleNotFoundError`, the wrong Python is being used — check with `which python` (macOS/Linux) or `where python` (Windows) that it points inside `.venv`.

## Module Overview

* **`pandas`**: Used for loading the raw `Survey_Results_UC.csv` dataset, handling missing values, and mapping ordinal Likert-scale strings to numerical integers.
* **`numpy`**: Provides support for efficient matrix operations and handling the adjacency matrix diagonals.
* **`scikit-learn`**: Utilized specifically for `sklearn.metrics.pairwise.cosine_similarity` to calculate the mathematical distance between respondent vectors.
* **`networkx`**: The core graph library used to construct the undirected network, filter isolated nodes, detect Louvain communities, and extract global/local metrics (density, centrality, modularity).
* **`matplotlib`**: Renders the 2D network visualizations using force-directed layouts (e.g., `spring_layout`).

