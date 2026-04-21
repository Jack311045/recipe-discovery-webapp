# Retrieval Module — Runbook

## Prerequisites

1. **Python 3.11+** installed
2. All dependencies installed: `pip install -r requirements.txt`
3. The embeddings and index artifacts exist at `data/artifacts/` including `recipe_embeddings.npy`, `recipe_ids.csv`, `recipe_index.joblib`
4. Run all commands from the **rsepository root** directory

---

## Step 0: Operate Pipeline

`python scripts/service.py`

What it does: 
1. Provides the k-nearest recipes by converting the embeddings into scores 
2. Employs cosine similarity
3. Ranks the scores
4. Implement constraints

## Step 1: Core Math Layer

`python scripts/similarity.py`

What it does: 
1. Takes an already converted array of query and processed CSV 
2. Compute cosine similarity between query vector and matrix of recipe embeddings
3. Normalize embeddings
4. Reduces similarity to dot products
5. Returns the similarity score for each recipe with the query 

## Step 2: Sorting Layer

`python scripts/ranker.py`

What it does: 
1. Makes a copy of the existing processed CSV
2. Adds a column of the scores associated with the specific recipe
3. Returns a sorted dataframe with the cosine similarity scores ranked from greatest to least  

## Step 3: Constraint Layer

`python scripts/filters.py`

What it does: 
1. Cyphers through the user query and obtains information on potential dietary restrictions, time limit, and ingredient limit
2. Performs a simple filtering based on the different restrictions given. 
3. Returns filtered dataframe

## Common Failure Cases and Mitigation Strategies

### 1. Missing Dependencies
**Failure Case:** The application fails to run due to missing Python packages.
**Mitigation:** Ensure all dependencies are listed in `requirements.txt` and run `pip install -r requirements.txt` before executing any scripts.

### 2. Incorrect File Paths
**Failure Case:** The application cannot find the required data files (e.g., `recipe_embeddings.npy`, `recipe_ids.csv`, `recipe_index.joblib`).
**Mitigation:** Verify that the files exist in the specified directory (`data/artifacts/`) and that the script is run from the repository root.

### 3. Invalid Input Data
**Failure Case:** The input data format is incorrect, leading to runtime errors.
**Mitigation:** Implement input validation checks to ensure that the data conforms to expected formats before processing.

### 4. Memory Issues
**Failure Case:** The application runs out of memory when processing large datasets.
**Mitigation:** Optimize data handling by using batch processing or reducing the size of the data being loaded into memory at once.

### 5. Runtime Errors in Scripts
**Failure Case:** Scripts may throw exceptions due to unforeseen issues in the code.
**Mitigation:** Wrap critical sections of code in try-except blocks and log errors for easier debugging.

### 6. Performance Bottlenecks
**Failure Case:** The application runs slowly due to inefficient algorithms.
**Mitigation:** Profile the code to identify bottlenecks and optimize algorithms, possibly using more efficient data structures or parallel processing.

### 7. User Input Errors
**Failure Case:** Users provide invalid or unexpected input during runtime.
**Mitigation:** Implement robust error handling and user prompts to guide users in providing correct input.

### 8. Dependency Version Conflicts
**Failure Case:** Conflicts arise from incompatible package versions.
**Mitigation:** Specify exact versions in `requirements.txt` and regularly update dependencies while testing for compatibility.

