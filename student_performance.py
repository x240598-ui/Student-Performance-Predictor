# ============================================================
# Student Performance Prediction — Linear Regression
# Dataset : Student_Performance.csv  (10,000 rows)
# Target  : Performance Index (0–100)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ------------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------
print("=" * 60)
print("STEP 1 : Loading dataset")
print("=" * 60)

df = pd.read_csv("Student_Performance.csv")
print(f"Shape  : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 : Exploratory Data Analysis")
print("=" * 60)

print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nStatistical summary:\n{df.describe()}")

# Missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing}")
if missing.sum() == 0:
    print(">> No missing values found.")
else:
    print(">> Dropping rows with missing values.")
    df.dropna(inplace=True)

# Duplicates
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows : {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f">> Duplicates removed. New shape: {df.shape}")
else:
    print(">> No duplicates found.")

# ------------------------------------------------------------------
# 3. Encode Categorical Feature
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 : Encoding categorical feature")
print("=" * 60)

# 'Extracurricular Activities' is Yes/No — map to 1/0
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
print("'Extracurricular Activities' encoded: Yes -> 1, No -> 0")
print(df["Extracurricular Activities"].value_counts())

# ------------------------------------------------------------------
# 4. Feature / Target Split
# ------------------------------------------------------------------
FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
TARGET = "Performance Index"

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeatures : {FEATURES}")
print(f"Target   : {TARGET}")

# ------------------------------------------------------------------
# 5. Train / Test Split  (80 / 20)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 : Train / Test Split (80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ------------------------------------------------------------------
# 6. Model Training
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 : Training Linear Regression model")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully.")
print(f"\nIntercept  : {model.intercept_:.4f}")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:45s}: {coef:.4f}")

# ------------------------------------------------------------------
# 7. Predictions & Evaluation
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6 : Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE  (Mean Absolute Error)       : {mae:.4f}")
print(f"MSE  (Mean Squared Error)        : {mse:.4f}")
print(f"RMSE (Root Mean Squared Error)   : {rmse:.4f}")
print(f"R2   (Coefficient of Determination): {r2:.4f}")

# ------------------------------------------------------------------
# 8. Visualisations
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7 : Generating graphs")
print("=" * 60)

sns.set_theme(style="whitegrid", palette="muted")

# --- Graph 1 : Hours Studied vs Performance Index ----------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Hours Studied"], df["Performance Index"],
           alpha=0.3, edgecolors="none", color="#3b82d4", s=15)
ax.set_xlabel("Hours Studied", fontsize=12)
ax.set_ylabel("Performance Index", fontsize=12)
ax.set_title("Hours Studied vs Performance Index", fontsize=14)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "hours_vs_performance.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"Saved : {path1}")

# --- Graph 2 : Previous Scores vs Performance Index --------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Previous Scores"], df["Performance Index"],
           alpha=0.3, edgecolors="none", color="#7c5cd8", s=15)
ax.set_xlabel("Previous Scores", fontsize=12)
ax.set_ylabel("Performance Index", fontsize=12)
ax.set_title("Previous Scores vs Performance Index", fontsize=14)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "scores_vs_performance.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"Saved : {path2}")

# --- Graph 3 : Actual vs Predicted Scores -------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.3, edgecolors="none",
           color="#22c55e", s=15, label="Predictions")
lims = [min(y_test.min(), y_pred.min()) - 2,
        max(y_test.max(), y_pred.max()) + 2]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Performance Index", fontsize=12)
ax.set_ylabel("Predicted Performance Index", fontsize=12)
ax.set_title("Actual vs Predicted Performance Index", fontsize=14)
ax.legend()
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
plt.savefig(path3, dpi=150)
plt.close()
print(f"Saved : {path3}")

# --- Graph 4 : Correlation Heatmap --------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(path4, dpi=150)
plt.close()
print(f"Saved : {path4}")

# ------------------------------------------------------------------
# 9. Save model
# ------------------------------------------------------------------
MODEL_PATH = "linear_regression_model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to : {MODEL_PATH}")

# ------------------------------------------------------------------
# 10. Quick sanity prediction
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8 : Sample prediction")
print("=" * 60)

sample = pd.DataFrame([[7, 80, 1, 7, 3]], columns=FEATURES)
predicted = model.predict(sample)[0]
print(f"Input  : Hours=7, PrevScore=80, Extracurricular=Yes, Sleep=7, Papers=3")
print(f"Predicted Performance Index : {predicted:.2f}")

print("\n" + "=" * 60)
print("All steps completed successfully!")
print("=" * 60)
