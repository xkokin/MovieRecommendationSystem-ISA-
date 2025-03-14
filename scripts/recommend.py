import pandas as pd
import tensorflow as tf
import numpy as np
import os

# paths
DATA_PATH = "../data/processed/movielens_1m_preprocessed.csv"
MODEL_PATH = "../models/recommender_model.h5"

# load preprocessed data
df = pd.read_csv(DATA_PATH, low_memory=False)

# load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Train the model first!")

model = tf.keras.models.load_model(MODEL_PATH)

# get unique user and movie ids
user_ids = df["UserID"].unique().tolist()
movie_ids = df["MovieID"].unique().tolist()

# create mappings to match training data
user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}
index_to_movie_id = {i: movie_id for movie_id, i in movie_id_to_index.items()}

# store movie titles for better recommendations
movie_id_to_title = df.set_index("MovieID")["Title"].to_dict()


# function to get top n recommendations
def get_top_n_recommendations(user_id, n=10):
    """returns top n recommended movies for a given user"""

    if user_id not in user_id_to_index:
        print(f"⚠️ User id {user_id} not found in dataset.")
        return []

    user_index = user_id_to_index[user_id]

    # get movies the user hasn't rated yet
    rated_movies = df[df["UserID"] == user_id]["MovieID"].values
    unrated_movie_ids = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]

    if not unrated_movie_ids:
        print(f"🎬 User {user_id} has rated all movies. No new recommendations.")
        return []

    # convert movie ids to index format for the model
    unrated_movie_indices = np.array([movie_id_to_index[movie_id] for movie_id in unrated_movie_ids])

    # predict ratings in one batch for efficiency
    user_indices = np.full_like(unrated_movie_indices, user_index)
    predicted_ratings = model.predict([user_indices, unrated_movie_indices]).flatten()

    # sort movies by predicted rating
    top_n_indices = np.argsort(predicted_ratings)[-n:][::-1]
    top_n_movies = [(index_to_movie_id[unrated_movie_indices[i]], predicted_ratings[i]) for i in top_n_indices]

    return top_n_movies


if __name__ == "__main__":
    try:
        # show valid user id range
        min_id, max_id = min(user_ids), max(user_ids)
        print(f"\n👥 Available user ids: {min_id} - {max_id}")

        user_id = int(input("enter userid: ").strip())

        recommendations = get_top_n_recommendations(user_id, 10)

        if recommendations:
            print("\n🎥 Top 10 movie recommendations:")
            for movie_id, rating in recommendations:
                movie_title = movie_id_to_title.get(movie_id, "unknown movie")  # handle missing titles safely
                print(f"⭐ {movie_title} (predicted rating: {rating:.2f})")

    except ValueError:
        print("❌ Invalid input! please enter a numeric userid.")