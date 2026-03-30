# Design Document

## Repo Structure

```
recipe-discovery-webapp/
├── README.md                          # Project overview, setup instructions, and usage guide
├── requirements.txt                   # Python dependencies with pinned minimum versions
├── pyproject.toml                     # Linter, formatter, and pytest configuration
├── Makefile                           # Shortcut commands for common tasks
├── .env.example                       # Template for environment variables
├── .gitignore                         # Git ignore rules for data, caches, and artifacts
│
├── app/                               # Streamlit frontend
│   ├── streamlit_app.py               # Main entry point and landing page
│   ├── pages/
│   │   ├── 1_Search.py                # Semantic search interface with filters
│   │   ├── 2_Explore_Clusters.py      # Cluster browsing view
│   │   ├── 3_Embedding_Map.py         # 2D scatter-plot of recipe embeddings
│   │   └── 4_Model_Diagnostics.py     # Model performance and evaluation metrics
│   └── components/
│       ├── filters.py                 # Reusable filter widgets (calories, tags, etc.)
│       ├── plots.py                   # Plotting helpers for Plotly charts
│       └── recipe_cards.py            # Recipe result card rendering
│
├── configs/                           # YAML configuration files
│   ├── base.yaml                      # Shared defaults (paths, random seed)
│   ├── data.yaml                      # Data cleaning and splitting parameters
│   ├── embeddings.yaml                # Encoder model name and batch size
│   ├── clustering.yaml                # Number of clusters, max iterations, tolerance
│   ├── regression.yaml                # Regression model hyperparameters
│   ├── reduction.yaml                 # PCA and autoencoder settings
│   └── retrieval.yaml                 # Top-K and similarity thresholds
│
├── data/                              # Local-only data storage (git-ignored)
│   ├── raw/                           # Original Kaggle CSV files
│   ├── interim/                       # Intermediate cleaned data
│   ├── processed/                     # Final train/val/test splits
│   └── artifacts/                     # Saved models, embeddings, and indices
│
├── docs/                              # Project documentation
│   ├── proposal.md                    # Written proposal (two paragraphs)
│   ├── design_doc.md                  # This document
│   └── architecture.md                # System architecture notes
│
├── notebooks/                         # Exploration and experimentation
│   ├── 01_data_audit.ipynb            # Initial data profiling and EDA
│   ├── 02_embedding_experiments.ipynb # Encoder comparison and embedding quality checks
│   ├── 03_clustering_and_regression.ipynb  # K-means and regression prototyping
│   └── 04_visualization.ipynb         # PCA and autoencoder projection experiments
│
├── scripts/                           # Runnable pipeline and training entrypoints
│   ├── run_data_pipeline.py           # End-to-end data cleaning and splitting
│   ├── build_embeddings.py            # Encode all recipes into dense vectors
│   ├── build_index.py                 # Build vector index for retrieval
│   ├── train_kmeans.py                # Fit k-means clusters on embeddings
│   ├── train_regression.py            # Train rating prediction model
│   ├── train_autoencoder.py           # Train autoencoder for 2D projection
│   ├── fit_pca.py                     # Fit PCA for 2D projection
│   ├── train_classifier.py            # Train optional tag classifier
│   └── evaluate_all.py               # Run evaluation metrics across all models
│
├── src/recipe_discovery/              # Main package source code
│   ├── settings.py                    # Project paths and configuration loader
│   ├── logging_utils.py               # Shared logging setup
│   ├── data/
│   │   ├── load.py                    # Load raw CSVs from disk
│   │   ├── preprocess.py              # Clean data and parse nutrition column
│   │   ├── corpus.py                  # Build recipe text representations for encoding
│   │   ├── schema.py                  # Column name constants and type definitions
│   │   └── splits.py                  # Train/val/test splitting logic
│   ├── embeddings/
│   │   ├── encoder.py                 # Sentence-transformer encoding wrapper
│   │   ├── index.py                   # Vector index construction and lookup
│   │   └── store.py                   # Save and load embedding matrices
│   ├── retrieval/
│   │   ├── similarity.py              # Cosine similarity computation
│   │   ├── filters.py                 # Nutritional and dietary constraint filters
│   │   ├── ranker.py                  # Combine similarity and regression scores
│   │   └── service.py                 # End-to-end retrieval pipeline
│   ├── clustering/
│   │   ├── kmeans.py                  # K-means implemented from scratch (no scikit-learn)
│   │   ├── labels.py                  # Cluster label analysis and naming
│   │   └── service.py                 # Clustering pipeline orchestration
│   ├── models/
│   │   ├── regression.py              # Rating prediction model
│   │   ├── classification.py          # Optional dietary tag classifier
│   │   └── inference.py               # Shared inference utilities
│   ├── reduction/
│   │   ├── pca.py                     # PCA projection to 2D
│   │   ├── autoencoder.py             # Deep autoencoder for nonlinear 2D projection
│   │   └── projection_service.py      # Unified interface for dimensionality reduction
│   ├── evaluation/
│   │   ├── metrics.py                 # Shared metric functions
│   │   ├── retrieval_eval.py          # Retrieval precision and recall
│   │   ├── clustering_eval.py         # Silhouette score and cluster quality
│   │   ├── regression_eval.py         # MSE and R² evaluation
│   │   └── classification_eval.py     # F1 and accuracy for tag prediction
│   └── utils/
│       ├── text.py                    # Text normalization helpers
│       ├── io.py                      # File I/O utilities
│       ├── seed.py                    # Reproducibility seed setter
│       └── validation.py              # Input validation helpers
│
└── tests/                             # Unit and smoke tests
    ├── test_preprocess.py             # Tests for data cleaning and parsing
    ├── test_kmeans.py                 # Tests for from-scratch k-means correctness
    ├── test_retrieval.py              # Tests for similarity search
    ├── test_regression.py             # Tests for rating predictor
    └── test_reduction.py              # Tests for PCA and autoencoder output shapes
```

## Division of Labor

| Member | Module | Deliverable |
|--------|--------|-------------|
| Aidan | Data pipeline and retrieval | Build a clean recipe corpus from Food.com metadata (`src/data/`, `scripts/run_data_pipeline.py`); use cosine similarity to retrieve semantically relevant recipes (`src/retrieval/`) |
| Jiaan Guo | Embeddings | Encode each recipe into a dense embedding using a pre-trained Transformer (`src/embeddings/`, `scripts/build_embeddings.py`, `scripts/build_index.py`) |
| Eddie | Clustering | Implement k-means clustering from scratch in PyTorch over recipe embeddings and integrate cluster outputs into the exploratory browsing view (`src/clustering/`, `scripts/train_kmeans.py`) |
| Blair | Regression | Fit a regression model for a rating-based ranking signal (`src/models/regression.py`, `scripts/train_regression.py`) |
| David | Dimensionality reduction and frontend | Project embeddings into 2D with PCA and a nonlinear autoencoder (`src/reduction/`, `scripts/fit_pca.py`, `scripts/train_autoencoder.py`); lead the Streamlit app integration (`app/`) |

All members jointly contribute to the final presentation and README documentation.
