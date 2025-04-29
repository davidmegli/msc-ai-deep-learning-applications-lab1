"""
Author: David Megli
Date: 2025-04-29
Description: Evaluate classical ML models on CNN features extracted from CIFAR-100.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os

# Path to extracted features file
feature_file = "cifar100_features.npz"

# Load features and labels
data = np.load(feature_file)
X = data["features"]
y = data["labels"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Results list
results = []

# Define classifiers
classifiers = {
    "Linear SVM": SVC(kernel="linear"),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB()
}

# Evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training and evaluating {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    results.append({"model": name, "accuracy": acc})

# Save results to CSV
output_path = "outputs/baseline_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")
