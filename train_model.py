# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=column_names, na_values="?")

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Fill missing values
df[['ca', 'thal']] = df[['ca', 'thal']].fillna(df[['ca', 'thal']].median())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Smart depth estimate
max_depth_value = int(np.log2(X_train.shape[0]))

# Train model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=max_depth_value,
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, "heart_model.pkl")

print("Model saved as heart_model.pkl")