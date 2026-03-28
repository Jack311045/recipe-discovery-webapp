# Architecture Notes

1. Build a clean recipe corpus from Food.com recipe metadata.
2. Encode each recipe into a dense embedding.
3. Use cosine similarity to retrieve semantically relevant recipes.
4. Fit k-means over recipe embeddings for exploratory browsing.
5. Fit a regression model for a rating-based ranking signal.
6. Project embeddings into 2D with PCA and a nonlinear autoencoder.
7. Serve search and exploration through a Streamlit app.
