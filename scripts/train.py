import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import os

# Load preprocessed data
data_path = "../data/processed/movielens_1m_preprocessed.csv"
df = pd.read_csv(data_path, low_memory=False)

# Define the Reader format for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["UserID", "MovieID", "Rating"]], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train a collaborative filtering model (SVD)
model = SVD()
model.fit(trainset)

# Save the trained model
model_path = "../models/"
os.makedirs(model_path, exist_ok=True)
with open(os.path.join(model_path, "recommender_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete! Model saved in 'models/'")
