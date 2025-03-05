
import pandas as pd
import os

# Load dataset paths
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"

users_file = os.path.join(RAW_DATA_DIR, "users.dat")
movies_file = os.path.join(RAW_DATA_DIR, "movies.dat")
ratings_file = os.path.join(RAW_DATA_DIR, "ratings.dat")

# Load datasets
users = pd.read_csv(users_file, sep="::", engine="python",
                    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])

movies = pd.read_csv(movies_file, sep="::", engine="python",
                     names=["MovieID", "Title", "Genres"], encoding="latin1")


ratings = pd.read_csv(ratings_file, sep="::", engine="python",
                      names=["UserID", "MovieID", "Rating", "Timestamp"])

# Preprocess
ratings["Timestamp"] = pd.to_datetime(ratings["Timestamp"], unit="s")
users["Gender"] = users["Gender"].map({"F": 0, "M": 1})

# Convert Genres to One-Hot Encoding
genres = movies["Genres"].str.get_dummies("|")
movies = pd.concat([movies.drop(columns=["Genres"]), genres], axis=1)

# Merge data
merged_df = ratings.merge(users, on="UserID").merge(movies, on="MovieID")

# Save preprocessed data
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
merged_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "movielens_1m_preprocessed.csv"), index=False)

print("Preprocessing complete! Saved in 'data/processed/'")