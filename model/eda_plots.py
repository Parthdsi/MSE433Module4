import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "MSE433_M4_combined_dataset.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["PT IN-OUT"])
df["DATE_parsed"] = pd.to_datetime(df["DATE"], errors="coerce")
df["case_of_day"] = df.groupby("DATE_parsed")["CASE #"].rank(method="first").astype("Int64")

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

COLORS = {"Dr. A": "#3b82f6", "Dr. B": "#ef4444", "Dr. C": "#10b981"}
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

# ── 1. Distribution of PT IN-OUT ──
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["PT IN-OUT"], bins=25, color="#6366f1", edgecolor="white", linewidth=0.8, alpha=0.85)
ax.axvline(df["PT IN-OUT"].mean(), color="#ef4444", linestyle="--", linewidth=2, label=f'Mean: {df["PT IN-OUT"].mean():.0f} min')
ax.axvline(df["PT IN-OUT"].median(), color="#f59e0b", linestyle="--", linewidth=2, label=f'Median: {df["PT IN-OUT"].median():.0f} min')
ax.set_xlabel("PT IN-OUT (minutes)")
ax.set_ylabel("Number of Cases")
ax.set_title("Distribution of Total Patient Time in Lab")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1_pt_inout_distribution.png"), dpi=150)
plt.close()
print("Saved: 1_pt_inout_distribution.png")

# ── 2. PT IN-OUT by Physician (box plot) ──
fig, ax = plt.subplots(figsize=(8, 5))
physicians = ["Dr. A", "Dr. B", "Dr. C"]
data_by_doc = [df[df["PHYSICIAN"] == doc]["PT IN-OUT"].values for doc in physicians]
bp = ax.boxplot(data_by_doc, labels=physicians, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
for patch, doc in zip(bp["boxes"], physicians):
    patch.set_facecolor(COLORS[doc])
    patch.set_alpha(0.7)
for i, doc in enumerate(physicians):
    vals = data_by_doc[i]
    ax.text(i + 1, vals.max() + 3, f"n={len(vals)}\nμ={vals.mean():.0f}",
            ha="center", fontsize=9, color="gray")
ax.set_ylabel("PT IN-OUT (minutes)")
ax.set_title("Case Duration by Physician")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "2_duration_by_physician.png"), dpi=150)
plt.close()
print("Saved: 2_duration_by_physician.png")

# ── 3. PT IN-OUT by Complexity Tier ──
fig, ax = plt.subplots(figsize=(8, 5))
tiers = ["Standard PVI", "BOX / PST BOX", "CTI / SVC", "AAFL"]
tier_colors = ["#6366f1", "#3b82f6", "#f59e0b", "#ef4444"]
data_by_tier = [df[df["complexity_label"] == t]["PT IN-OUT"].values for t in tiers]
bp = ax.boxplot(data_by_tier, labels=tiers, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
for patch, c in zip(bp["boxes"], tier_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
for i, t in enumerate(tiers):
    vals = data_by_tier[i]
    if len(vals) > 0:
        ax.text(i + 1, vals.max() + 3, f"n={len(vals)}\nμ={vals.mean():.0f}",
                ha="center", fontsize=9, color="gray")
ax.set_ylabel("PT IN-OUT (minutes)")
ax.set_title("Case Duration by Procedure Complexity")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "3_duration_by_complexity.png"), dpi=150)
plt.close()
print("Saved: 3_duration_by_complexity.png")

# ── 4. PT IN-OUT by Case Position in Day ──
fig, ax = plt.subplots(figsize=(8, 5))
df_valid = df.dropna(subset=["case_of_day"])
positions = sorted(df_valid["case_of_day"].unique())
data_by_pos = [df_valid[df_valid["case_of_day"] == p]["PT IN-OUT"].values for p in positions]
pos_labels = [f"Case {int(p)}" for p in positions]
bp = ax.boxplot(data_by_pos, labels=pos_labels, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("#6366f1")
    patch.set_alpha(0.5 + 0.07 * i)
for i, p in enumerate(positions):
    vals = data_by_pos[i]
    if len(vals) > 0:
        ax.text(i + 1, vals.max() + 3, f"n={len(vals)}", ha="center", fontsize=9, color="gray")
ax.set_ylabel("PT IN-OUT (minutes)")
ax.set_title("Case Duration by Position in Day")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "4_duration_by_case_position.png"), dpi=150)
plt.close()
print("Saved: 4_duration_by_case_position.png")

# ── 5. Correlation heatmap of timing variables ──
timing_cols = ["PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
               "ABL DURATION", "ABL TIME", "#ABL", "LA DWELL TIME",
               "CASE TIME", "POST CARE/EXTUBATION", "PT IN-OUT"]
corr = df[timing_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(timing_cols)))
ax.set_yticks(range(len(timing_cols)))
short_labels = ["Prep", "Access", "TSP", "Pre-Map", "ABL Dur",
                "ABL Time", "#ABL", "LA Dwell", "Case Time", "Post Care", "PT IN-OUT"]
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short_labels, fontsize=9)
for i in range(len(timing_cols)):
    for j in range(len(timing_cols)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
ax.set_title("Correlation Between Timing Variables")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "5_correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: 5_correlation_heatmap.png")

# ── 6. Physician × Complexity breakdown (grouped bar) ──
fig, ax = plt.subplots(figsize=(9, 5))
tiers = ["Standard PVI", "BOX / PST BOX", "CTI / SVC", "AAFL"]
x = np.arange(len(tiers))
width = 0.25
for i, doc in enumerate(physicians):
    means = []
    for t in tiers:
        subset = df[(df["PHYSICIAN"] == doc) & (df["complexity_label"] == t)]["PT IN-OUT"]
        means.append(subset.mean() if len(subset) > 0 else 0)
    bars = ax.bar(x + i * width, means, width, label=doc, color=COLORS[doc], alpha=0.8, edgecolor="white")
    for bar, m in zip(bars, means):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{m:.0f}", ha="center", fontsize=8, color="gray")
ax.set_xticks(x + width)
ax.set_xticklabels(tiers)
ax.set_ylabel("Mean PT IN-OUT (minutes)")
ax.set_title("Average Duration by Physician × Complexity")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "6_physician_complexity.png"), dpi=150)
plt.close()
print("Saved: 6_physician_complexity.png")

# ── 7. Time breakdown stacked bar (what makes up PT IN-OUT) ──
fig, ax = plt.subplots(figsize=(9, 5))
components = ["PT PREP/INTUBATION", "ACCESSS", "TSP", "PRE-MAP",
              "ABL DURATION", "POST CARE/EXTUBATION"]
comp_labels = ["Prep/Intubation", "Access", "TSP", "Pre-Map", "Ablation", "Post Care"]
comp_colors = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
means_by_doc = {}
for doc in physicians:
    subset = df[df["PHYSICIAN"] == doc]
    means_by_doc[doc] = [subset[c].mean() for c in components]

x = np.arange(len(physicians))
bottom = np.zeros(len(physicians))
for j, (comp, label, color) in enumerate(zip(components, comp_labels, comp_colors)):
    vals = [means_by_doc[doc][j] for doc in physicians]
    ax.bar(x, vals, 0.5, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(vals):
        if v > 4:
            ax.text(i, bottom[i] + v/2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottom += vals

for i, doc in enumerate(physicians):
    ax.text(i, bottom[i] + 2, f"Total: {sum(means_by_doc[doc]):.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(physicians)
ax.set_ylabel("Minutes")
ax.set_title("Average Procedure Time Breakdown by Physician")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "7_time_breakdown.png"), dpi=150)
plt.close()
print("Saved: 7_time_breakdown.png")

# ── 8. Cases over time (scatter colored by physician) ──
fig, ax = plt.subplots(figsize=(10, 5))
df_dated = df.dropna(subset=["DATE_parsed"])
for doc in physicians:
    subset = df_dated[df_dated["PHYSICIAN"] == doc]
    ax.scatter(subset["DATE_parsed"], subset["PT IN-OUT"], c=COLORS[doc],
               label=doc, alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
ax.set_xlabel("Date")
ax.set_ylabel("PT IN-OUT (minutes)")
ax.set_title("Case Durations Over Time")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "8_duration_over_time.png"), dpi=150)
plt.close()
print("Saved: 8_duration_over_time.png")

# ── 9. Pre-case flags impact (Model 2 features) ──
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
flags = [
    ("obesity", "Obesity"),
    ("sleep_apnea", "Sleep Apnea"),
    ("fasting_not_confirmed", "Fasting Not Confirmed"),
    ("bloodwork_incomplete", "Bloodwork Incomplete"),
    ("equipment_prestaged", "Equipment Pre-staged"),
    ("anesthesia_ready", "Anesthesia Ready"),
]
for ax, (col, label) in zip(axes.flat, flags):
    yes = df[df[col] == "Y"]["PT IN-OUT"]
    no = df[df[col] == "N"]["PT IN-OUT"]
    bp = ax.boxplot([no.values, yes.values], labels=["No", "Yes"], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#94a3b8")
    bp["boxes"][1].set_facecolor("#f59e0b")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel("PT IN-OUT (min)", fontsize=8)
    ax.text(1, no.max() + 5, f"n={len(no)}", ha="center", fontsize=8, color="gray")
    ax.text(2, yes.max() + 5, f"n={len(yes)}", ha="center", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Impact of Pre-Case Flags on Duration (Model 2 Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "9_precase_flags_impact.png"), dpi=150)
plt.close()
print("Saved: 9_precase_flags_impact.png")

print(f"\nAll plots saved to: {OUT_DIR}/")
