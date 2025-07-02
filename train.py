import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Load
df = pd.read_csv("heart.csv")        # or heart_cleveland_upload.csv

# 2. Features / target
X = df.drop("condition", axis=1)
y = df["condition"]

# 3. Train
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="liblinear")
)
model.fit(X, y)

# 4. Persist
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved")
