import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split
import os

# Load preprocessed data
data_path = "../data/processed/movielens_1m_preprocessed.csv"
df = pd.read_csv(data_path, low_memory=False)

# Extract unique user and movie IDs
user_ids = df["UserID"].unique().tolist()
movie_ids = df["MovieID"].unique().tolist()

# Create mapping from IDs to index values
user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}

# Convert UserID and MovieID to indexed values
df["UserID"] = df["UserID"].map(user_id_to_index)
df["MovieID"] = df["MovieID"].map(movie_id_to_index)


# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_size = 50

# Define the Neural Collaborative Filtering (NCF) Model
input_user = Input(shape=(1,))
input_movie = Input(shape=(1,))

# Embeddings
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(input_user)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(input_movie)

# Flatten embeddings
user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

# Concatenate and pass through dense layers
concat = Concatenate()([user_vec, movie_vec])
dense1 = Dense(128, activation="relu")(concat)
dense2 = Dense(64, activation="relu")(dense1)
output = Dense(1, activation="linear")(dense2)  # Predict a rating


# Create and compile model
model = keras.Model(inputs=[input_user, input_movie], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Convert data to NumPy arrays
train_X = [train["UserID"].values, train["MovieID"].values]
train_y = train["Rating"].values
test_X = [test["UserID"].values, test["MovieID"].values]
test_y = test["Rating"].values

# Train the model
model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y))

# Save the model
model_path = "../models/"
os.makedirs(model_path, exist_ok=True)
model.save(os.path.join(model_path, "recommender_model.h5"))

print("Model training complete! Model saved in 'models/'")
