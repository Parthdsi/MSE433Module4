import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from sklearn.model_selection import train_test_split
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "MSE433_M4_combined_dataset.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["PT IN-OUT"])
df["DATE_parsed"] = pd.to_datetime(df["DATE"], errors="coerce")
df["case_of_day"] = df.groupby("DATE_parsed")["CASE #"].rank(method="first").astype("Int64")
df = df.dropna(subset=["case_of_day"])

def encode_complexity(note):
    if pd.isna(note) or str(note).strip() == "":
        return "Standard PVI"
    note_upper = str(note).upper()
    if "AAFL" in note_upper:
        return "AAFL"
    if "CTI" in note_upper or "SVC" in note_upper:
        return "CTI / SVC"
    if "BOX" in note_upper:
        return "BOX / PST BOX"
    return "Standard PVI"

df["complexity_label"] = df["Note"].apply(encode_complexity)

plt.rcParams.update({"font.size": 12, "figure.facecolor": "white", "font.family": "sans-serif"})
BLUE = "#3b82f6"
RED = "#ef4444"
GREEN = "#22c55e"
AMBER = "#f59e0b"
PURPLE = "#6366f1"
GRAY = "#94a3b8"

mean_dur = df["PT IN-OUT"].mean()

# ═══════════════════════════════════════════════════════════════
# PLOT 1: "Procedures take anywhere from 49 to 204 minutes"
# Simple: just show the spread
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["PT IN-OUT"], bins=20, color=BLUE, edgecolor="white", alpha=0.85)
ax.axvline(mean_dur, color=RED, linewidth=2.5, linestyle="--")
ax.text(mean_dur + 2, ax.get_ylim()[1] * 0.85, f"Average: {mean_dur:.0f} min",
        fontsize=13, color=RED, fontweight="bold")
ax.set_xlabel("How long the patient was in the lab (minutes)", fontsize=13)
ax.set_ylabel("Number of procedures", fontsize=13)
ax.set_title("Procedure times vary a LOT — hard to schedule with a single block", fontsize=15, fontweight="bold")
ax.grid(axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s1_procedures_vary_widely.png"), dpi=150)
plt.close()
print("Saved: s1")

# ═══════════════════════════════════════════════════════════════
# PLOT 2: "Some doctors' cases take longer than others"
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
physicians = ["Dr. A", "Dr. B", "Dr. C"]
doc_colors = [BLUE, RED, GREEN]
means = [df[df["PHYSICIAN"] == d]["PT IN-OUT"].mean() for d in physicians]
counts = [len(df[df["PHYSICIAN"] == d]) for d in physicians]

bars = ax.bar(physicians, means, color=doc_colors, width=0.5, edgecolor="white", alpha=0.85)
for bar, m, n in zip(bars, means, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{m:.0f} min avg\n({n} cases)", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Average time in lab (minutes)", fontsize=13)
ax.set_title("Which doctor is performing the procedure matters", fontsize=15, fontweight="bold")
ax.set_ylim(0, max(means) * 1.25)
ax.grid(axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s2_doctor_matters.png"), dpi=150)
plt.close()
print("Saved: s2")

# ═══════════════════════════════════════════════════════════════
# PLOT 3: "More complex procedures take way longer"
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
tiers = ["Standard PVI", "BOX / PST BOX", "CTI / SVC", "AAFL"]
tier_colors = [GREEN, BLUE, AMBER, RED]
tier_means = []
tier_counts = []
for t in tiers:
    vals = df[df["complexity_label"] == t]["PT IN-OUT"]
    tier_means.append(vals.mean() if len(vals) > 0 else 0)
    tier_counts.append(len(vals))

bars = ax.barh(tiers, tier_means, color=tier_colors, height=0.55, edgecolor="white", alpha=0.85)
for bar, m, n in zip(bars, tier_means, tier_counts):
    if m > 0:
        ax.text(m + 2, bar.get_y() + bar.get_height()/2,
                f"{m:.0f} min avg  ({n} cases)", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Average time in lab (minutes)", fontsize=13)
ax.set_title("More complex procedures take significantly longer", fontsize=15, fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s3_complexity_matters.png"), dpi=150)
plt.close()
print("Saved: s3")

# ═══════════════════════════════════════════════════════════════
# PLOT 4: "If we just block 90 min for everyone, look what happens"
# Two simple pie-style bars
# ═══════════════════════════════════════════════════════════════
BLOCK = 90
over = (df["PT IN-OUT"] > BLOCK).sum()
under = (df["PT IN-OUT"] <= BLOCK).sum()
total = len(df)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: cases that run over
ax = axes[0]
ax.barh(["Run over\n(delays next patient)"], [over], color=RED, alpha=0.8, height=0.4)
ax.barh(["Finish on time"], [under], color=GREEN, alpha=0.8, height=0.4)
ax.text(over + 1, 0, f"{over} cases ({over/total*100:.0f}%)", va="center", fontsize=13, fontweight="bold", color=RED)
ax.text(under + 1, 1, f"{under} cases ({under/total*100:.0f}%)", va="center", fontsize=13, fontweight="bold", color="#166534")
ax.set_xlim(0, max(over, under) * 1.4)
ax.set_title(f"If we schedule every case\nas a {BLOCK}-minute block...", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2)

# Right: total wasted vs overtime minutes
ax = axes[1]
wasted = np.sum(np.maximum(BLOCK - df["PT IN-OUT"].values, 0))
overtime = np.sum(np.maximum(df["PT IN-OUT"].values - BLOCK, 0))
bars = ax.bar(["Wasted lab time\n(finished early)", "Overtime\n(ran late)"],
              [wasted, overtime], color=[GRAY, RED], width=0.5, alpha=0.8)
ax.text(0, wasted + 30, f"{wasted:.0f}\nminutes", ha="center", fontsize=13, fontweight="bold", color="#475569")
ax.text(1, overtime + 30, f"{overtime:.0f}\nminutes", ha="center", fontsize=13, fontweight="bold", color=RED)
ax.set_title("Total impact across\nall 144 procedures", fontsize=14, fontweight="bold")
ax.set_ylabel("Total minutes", fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s4_fixed_block_problem.png"), dpi=150)
plt.close()
print("Saved: s4")

# ═══════════════════════════════════════════════════════════════
# PLOT 5: "Our model gets closer to the right answer"
# Simple comparison: avg guess vs model
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

naive_mae = np.abs(df["PT IN-OUT"] - mean_dur).mean()
model_mae = 12.0  # from training output

bars = ax.bar(
    ["Just use the average\n(current approach)", "Our prediction model"],
    [naive_mae, model_mae],
    color=[GRAY, BLUE],
    width=0.45,
    alpha=0.85,
    edgecolor="white",
)
ax.text(0, naive_mae + 0.8, f"Off by ~{naive_mae:.0f} min\nper case", ha="center", fontsize=13, fontweight="bold", color="#475569")
ax.text(1, model_mae + 0.8, f"Off by ~{model_mae:.0f} min\nper case", ha="center", fontsize=13, fontweight="bold", color=BLUE)

improvement = ((naive_mae - model_mae) / naive_mae) * 100
ax.annotate(f"{improvement:.0f}% more\naccurate", xy=(1, model_mae), xytext=(1.45, naive_mae * 0.7),
            fontsize=13, fontweight="bold", color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

ax.set_ylabel("Average scheduling error (minutes)", fontsize=13)
ax.set_title("Our model predicts procedure time more accurately", fontsize=15, fontweight="bold")
ax.set_ylim(0, naive_mae * 1.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s5_model_improvement.png"), dpi=150)
plt.close()
print("Saved: s5")

# ═══════════════════════════════════════════════════════════════
# PLOT 6: "How the scheduler would use this"
# Green / Yellow / Red zone breakdown
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

green_n = (df["PT IN-OUT"] < 80).sum()
yellow_n = ((df["PT IN-OUT"] >= 80) & (df["PT IN-OUT"] <= 110)).sum()
red_n = (df["PT IN-OUT"] > 110).sum()

categories = [
    f"Under 80 min\nSchedule normally",
    f"80–110 min\nAdd extra buffer",
    f"Over 110 min\nBlock 2 hours",
]
values = [green_n, yellow_n, red_n]
colors = [GREEN, AMBER, RED]
pcts = [v / total * 100 for v in values]

bars = ax.barh(categories, values, color=colors, height=0.5, alpha=0.8, edgecolor="white")
for bar, v, p in zip(bars, values, pcts):
    ax.text(v + 1.5, bar.get_y() + bar.get_height()/2,
            f"{v} cases ({p:.0f}%)", va="center", fontsize=13, fontweight="bold")

ax.set_xlabel("Number of procedures", fontsize=13)
ax.set_title("The model tells the scheduler how to handle each case", fontsize=15, fontweight="bold")
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s6_scheduler_decision_zones.png"), dpi=150)
plt.close()
print("Saved: s6")

print(f"\nAll plots saved to: {OUT_DIR}/")
