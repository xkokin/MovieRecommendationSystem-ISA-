import pandas as pd
import pickle
import os
from surprise import Reader, Dataset

# Paths
data_path = "../data/processed/movielens_1m_preprocessed.csv"
model_path = "../models/recommender_model.pkl"

# Load preprocessed data
df = pd.read_csv(data_path, low_memory=False)

# Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)


# Function to get top N recommendations
def get_top_n_recommendations(user_id, n=10):
    unique_movie_ids = df["MovieID"].unique()
    rated_movies = df[df["UserID"] == user_id]["MovieID"].values
    unrated_movies = [movie_id for movie_id in unique_movie_ids if movie_id not in rated_movies]

    predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = predictions[:n]
    return top_n


# Example usage
if __name__ == "__main__":
    user_id = int(input("Enter UserID: "))
    recommendations = get_top_n_recommendations(user_id, 10)

    print("Top 10 movie recommendations:")
    for movie_id, rating in recommendations:
        movie_title = df[df["MovieID"] == movie_id]["Title"].values[0]
        print(f"{movie_title} (Predicted Rating: {rating:.2f})")