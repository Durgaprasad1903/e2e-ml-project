import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="species"
)

y = y.replace({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

df = pd.concat([X, y], axis=1)

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(6, 4))
sns.histplot(df["sepal length (cm)"])
plt.title("Sepal Length Distribution")
plt.savefig("sepal_length_distribution.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x="species", data=df)
plt.title("Species Distribution")
plt.savefig("species_distribution.png")
plt.close()

print("Plots saved successfully.")

# -----------------------------
# Missing values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Scaling
# -----------------------------

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA
# -----------------------------

pca = PCA(n_components=0.99)

X_pca = pca.fit_transform(X_scaled)

print("\nOriginal Features:", X.shape[1])
print("Features After PCA:", X_pca.shape[1])

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# -----------------------------
# Cross Validation
# -----------------------------

def cv_score(model):
    return cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    ).mean() * 100


print("\nCross Validation Scores")

print(
    "Logistic Regression:",
    round(cv_score(LogisticRegression(max_iter=1000)), 2)
)

print(
    "Random Forest:",
    round(cv_score(RandomForestClassifier(random_state=42)), 2)
)

print(
    "SVM:",
    round(cv_score(SVC()), 2)
)

# -----------------------------
# Hyperparameter Tuning
# -----------------------------

grid = GridSearchCV(
    SVC(),
    {
        "C": [0.5, 1, 2, 3],
        "kernel": ["linear", "rbf"]
    },
    cv=5
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

# -----------------------------
# Final Model
# -----------------------------

model = SVC(
    C=grid.best_params_["C"],
    kernel=grid.best_params_["kernel"]
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100

print("\nTest Accuracy:", round(accuracy, 2), "%")

# -----------------------------
# Prediction
# -----------------------------

pred = model.predict(X_test)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        pred
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cf = confusion_matrix(
    y_test,
    pred,
    normalize="true"
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cf,
    annot=True,
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.close()

print("\nConfusion matrix saved successfully.")

# -----------------------------
# Save Objects
# -----------------------------

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("\nModel saved as model.pkl")
print("Scaler saved as scaler.pkl")
print("PCA saved as pca.pkl")