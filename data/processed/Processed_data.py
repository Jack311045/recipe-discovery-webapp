import numpy as np
import pandas as pd

# Load the CSV file
data = pd.read_csv('data/raw/RAW_recipes.csv')
data = data[data['tags'].notnull()]
# Convert the tags column from string representation of lists to actual lists
data['tags'] = data['tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)
print('hi')

# Create one-hot encoding for the tags
tags_one_hot = data['tags'].explode().str.get_dummies()
tags_one_hot = tags_one_hot.sum(axis=0).to_frame().T

# Concatenate the one-hot encoded tags with the original dataframe
data = pd.concat([data, tags_one_hot], axis=1)

# Convert the nutrition column from string representation of lists to actual lists
data['nutrition'] = data['nutrition'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Create new columns for each nutritional component
nutrition_columns = ['calories', 'total fat', 'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates']
data[nutrition_columns] = pd.DataFrame(data['nutrition'].tolist(), index=data.index)

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

# Print out the number of nulls that exist per column
null_values = data.isnull().sum()
# Export the dataframe to an Excel file
data.to_excel('Processed_data.xlsx', index=False)
