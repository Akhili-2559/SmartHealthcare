import pickle
print("Loading model...")
with open("models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")