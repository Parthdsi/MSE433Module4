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

# --- Derive features ---

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

# case_of_day
df["DATE_parsed"] = pd.to_datetime(df["DATE"], errors="coerce")
df["case_of_day"] = df.groupby("DATE_parsed")["CASE #"].rank(method="first").astype("Int64")

# Binary encodings for app-logged fields
yn_cols = {
    "first_case_of_day": "first_case_of_day",
    "obesity": "obesity",
    "sleep_apnea": "sleep_apnea",
    "fasting_not_confirmed": "fasting_not_confirmed",
    "bloodwork_incomplete": "bloodwork_incomplete",
    "equipment_prestaged": "equipment_prestaged",
    "anesthesia_ready": "anesthesia_ready",
}

for feat_name, col_name in yn_cols.items():
    df[feat_name + "_enc"] = df[col_name].map({"Y": 1, "N": 0}).fillna(0).astype(int)

# Drop rows where PT IN-OUT is null
df = df.dropna(subset=["PT IN-OUT"])
df = df.dropna(subset=["case_of_day"])

print(f"Dataset size after cleaning: {len(df)} rows")

# One-hot encode PHYSICIAN (drop_first=True)
physician_dummies = pd.get_dummies(df["PHYSICIAN"], prefix="physician", drop_first=True)

features = pd.concat([
    physician_dummies,
    df[["case_of_day", "complexity_tier"]],
    df[[c + "_enc" for c in yn_cols.keys()]],
], axis=1).astype(float)

FEATURE_NAMES = list(features.columns)
print(f"Features: {FEATURE_NAMES}")

target = df["PT IN-OUT"].values
X = features.values

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

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

model_path = os.path.join(SCRIPT_DIR, "ep_model2.pkl")
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
ax.set_title(f"Model 2 — {best_name}\nMAE={best['mae']:.1f} min  |  R²={best['r2']:.3f}", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "model2_performance.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")

# --- Side-by-side comparison with Model 1 ---
print(f"\n{'='*55}")
print(f"{'COMPARISON':^55}")
print(f"{'='*55}")

m1_path = os.path.join(SCRIPT_DIR, "ep_model1.pkl")
if os.path.exists(m1_path):
    m1_data = joblib.load(m1_path)
    m1_model = m1_data["model"]
    m1_features = m1_data["feature_names"]
    m1_name = m1_data["model_name"]

    # Rebuild Model 1 features for comparison on same test split
    df_full = pd.read_csv(CSV_PATH)
    df_full["complexity_tier"] = df_full["Note"].apply(encode_complexity)
    df_full["DATE_parsed"] = pd.to_datetime(df_full["DATE"], errors="coerce")
    df_full["case_of_day"] = df_full.groupby("DATE_parsed")["CASE #"].rank(method="first").astype("Int64")
    df_full = df_full.dropna(subset=["PT IN-OUT", "case_of_day"])

    physician_dummies_full = pd.get_dummies(df_full["PHYSICIAN"], prefix="physician", drop_first=True)
    m1_feats = pd.concat([
        physician_dummies_full,
        df_full[["case_of_day", "complexity_tier"]],
    ], axis=1).astype(float)

    m1_X = m1_feats.values
    m1_y = df_full["PT IN-OUT"].values
    _, m1_X_test, _, m1_y_test = train_test_split(m1_X, m1_y, test_size=0.2, random_state=42)

    m1_pred = m1_model.predict(m1_X_test)
    m1_mae = mean_absolute_error(m1_y_test, m1_pred)
    m1_r2 = r2_score(m1_y_test, m1_pred)

    print(f"{'Metric':<20} {'Model 1':>15} {'Model 2':>15}")
    print(f"{'-'*50}")
    print(f"{'Best Algorithm':<20} {m1_name:>15} {best_name:>15}")
    print(f"{'MAE (min)':<20} {m1_mae:>15.2f} {best['mae']:>15.2f}")
    print(f"{'R²':<20} {m1_r2:>15.3f} {best['r2']:>15.3f}")
    improvement = ((m1_mae - best["mae"]) / m1_mae) * 100
    print(f"{'MAE Improvement':<20} {'—':>15} {improvement:>14.1f}%")
else:
    print("Model 1 not found — run train_model1.py first for comparison.")

print(f'\n"Model 2 uses simulated app data. Accuracy figures are illustrative — real performance will be established after app deployment."')
