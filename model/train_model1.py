import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "MSE433_M4_combined_dataset.csv")

df = pd.read_csv(CSV_PATH)

# --- Derive features from original hospital data only ---

# complexity_tier from Note column
def encode_complexity(note):
    if pd.isna(note) or str(note).strip() == "":
        return 0
    note_upper = str(note).upper()
    if "AAFL" in note_upper:
        return 3
    if "CTI" in note_upper or "SVC" in note_upper:
        return 2
    if "BOX" in note_upper:
        return 1
    return 0

df["complexity_tier"] = df["Note"].apply(encode_complexity)

# case_of_day: rank each case's position within its DATE group
df["DATE_parsed"] = pd.to_datetime(df["DATE"], errors="coerce")
df["case_of_day"] = df.groupby("DATE_parsed")["CASE #"].rank(method="first").astype("Int64")

# Drop rows where PT IN-OUT is null
df = df.dropna(subset=["PT IN-OUT"])
# Also drop rows where case_of_day couldn't be computed (bad dates)
df = df.dropna(subset=["case_of_day"])

print(f"Dataset size after cleaning: {len(df)} rows")
print(f"\n--- Baseline ---")
print(f"PT IN-OUT mean: {df['PT IN-OUT'].mean():.1f} min")
print(f"PT IN-OUT std:  {df['PT IN-OUT'].std():.1f} min")
print(f"PT IN-OUT range: {df['PT IN-OUT'].min():.0f} – {df['PT IN-OUT'].max():.0f} min")

# One-hot encode PHYSICIAN (drop_first=True)
physician_dummies = pd.get_dummies(df["PHYSICIAN"], prefix="physician", drop_first=True)

features = pd.concat([
    physician_dummies,
    df[["case_of_day", "complexity_tier"]],
], axis=1).astype(float)

FEATURE_NAMES = list(features.columns)
print(f"\nFeatures: {FEATURE_NAMES}")

target = df["PT IN-OUT"].values
X = features.values

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 52)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}
    print(f"{name:<25} {mae:>8.2f} {rmse:>8.2f} {r2:>8.3f}")

best_name = min(results, key=lambda k: results[k]["mae"])
best = results[best_name]
print(f"\nBest model: {best_name} (MAE={best['mae']:.2f})")

model_path = os.path.join(SCRIPT_DIR, "ep_model1.pkl")
joblib.dump({
    "model": best["model"],
    "feature_names": FEATURE_NAMES,
    "model_name": best_name,
}, model_path)
print(f"Saved: {model_path}")

# Feature importances
print(f"\n--- Feature Importances ({best_name}) ---")
if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
elif hasattr(best["model"], "coef_"):
    importances = np.abs(best["model"].coef_)
else:
    importances = np.zeros(len(FEATURE_NAMES))

for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
    print(f"  {fname:<25} {imp:.4f}")

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best["y_pred"], alpha=0.7, edgecolors="k", linewidths=0.5)
lims = [
    min(min(y_test), min(best["y_pred"])) - 5,
    max(max(y_test), max(best["y_pred"])) + 5,
]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual PT IN-OUT (min)", fontsize=12)
ax.set_ylabel("Predicted PT IN-OUT (min)", fontsize=12)
ax.set_title(f"Model 1 — {best_name}\nMAE={best['mae']:.1f} min  |  R²={best['r2']:.3f}", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "model1_performance.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")

print(f'\n"Model 1 uses only data available today. To improve accuracy, deploy the logging app to capture patient complexity flags."')
