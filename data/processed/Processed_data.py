import numpy as np
import pandas as pd
import sys
import subprocess

# Load the CSV file
data = pd.read_csv('data/raw/RAW_recipes.csv')
data = data[data['tags'].notnull()]

# Replace multiple spaces with a single space in the 'name' column
data['name'] = data['name'].str.replace(r'\s{2,}', ' ', regex=True)

# Replace hyphens with spaces in each tag
data['tags'] = data['tags'].apply(lambda x: [tag.replace('-', ' ') for tag in x] if isinstance(x, list) else x)

# Convert the tags column from string representation of lists to actual lists
data['tags'] = data['tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)
print('hi')

# Ensure 'steps' column is a list and check 'n_steps' consistency
data['steps'] = data['steps'].apply(lambda x: eval(x) if isinstance(x, str) else x)
data['n_steps'] = data.apply(lambda row: len(row['steps']) if len(row['steps']) != row['n_steps'] else row['n_steps'], axis=1)

# Find all unique items in all lists across the tags column
tag_counts = data['tags'].explode().value_counts()
filtered_tags = tag_counts[tag_counts >= 10].index.tolist()
unique_items = sorted(set(filtered_tags))
unique_items = [item for item in unique_items if item != 'less_thansql:name_topics_of_recipegreater_than' and item != 'free-of-something' and item != 'Throw the ultimate fiesta with this sopaipillas recipe from Food.com.' and item != 'number-of-servings']

# Create one-hot encoding for all unique items
one_hot_encoded = pd.DataFrame({item: data['tags'].apply(lambda x: 1 if item in x else 0) for item in unique_items})
data = pd.concat([data, one_hot_encoded], axis=1)

# Convert the nutrition column from string representation of lists to actual lists
data['nutrition'] = data['nutrition'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Create new columns for each nutritional component
nutrition_columns = ['calories', 'total fat', 'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates']
nutrition_data = pd.DataFrame(data['nutrition'].tolist(), columns=nutrition_columns, index=data.index)
data = pd.concat([data, nutrition_data], axis=1)

# Rename 'id' to 'recipe_id'
data.rename(columns={'id': 'recipe_id'}, inplace=True)

# Scan for any null values and remove them
data = data.dropna()

# Remove the 'contribution id' and 'submission date' columns
data = data.drop(columns=['contributor_id', 'submitted'], errors='ignore')

# Ensure specific columns are of float type
float_columns = ['recipe_id', 'calories', 'total fat', 'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates', 'n_steps', 'minutes']
data[float_columns] = data[float_columns].astype(float)

# Ensure the rest of the columns are of string type
string_columns = data.columns.difference(float_columns)
data[string_columns] = data[string_columns].astype(str)

# Load the interactions CSV file
interactions = pd.read_csv('data/raw/RAW_interactions.csv')
interactions = interactions.drop(columns=['date'], errors='ignore')

# Group by 'id' to calculate the average rating and aggregate reviews
average_ratings = interactions.groupby('recipe_id').agg(
    rating=('rating', 'mean'),
    num_ratings=('rating', 'count'),  # Count of ratings
    all_reviews=('review', lambda x: list(x)),
).reset_index()
average_ratings = average_ratings.rename(columns={'id': 'recipe_id'})

# Ensure rating and num_ratings are integers and all_reviews are lists of strings
average_ratings['rating'] = average_ratings['rating'].astype(int)
average_ratings['num_ratings'] = average_ratings['num_ratings'].astype(int)
average_ratings['all_reviews'] = average_ratings['all_reviews'].apply(lambda x: [str(review) for review in x])

# Merge the average ratings and reviews back into the original data
data = data.merge(average_ratings, on='recipe_id', how='left')

# Reorder the dataframe based on the recipe_id column
data = data.sort_values(by='recipe_id').reset_index(drop=True)
# Remove the 'tags' and 'nutrition' columns
data = data.drop(columns=['tags', 'nutrition'], errors='ignore')

data.to_csv('Processed_data_updated2.csv', index=False)
print('done')
