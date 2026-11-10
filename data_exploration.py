import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

OUT = "eda_plots"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
SCORE_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.png")


# Load data and split into labeled / unlabeled
df = pd.read_csv("data/train.csv")
lab = df[df.Score.notna()].copy()
unl = df[df.Score.isna()].copy()
lab["Score"] = lab.Score.astype(int)
print(f"\nrows={len(df):,}  labeled={len(lab):,}  unlabeled={len(unl):,}")
print("\nDtypes:\n", df.dtypes.to_string())
print("\nMissing values (%):\n", (df.isna().mean() * 100).round(2).to_string())

# 1. Missing values chart
miss = (df.isna().mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(miss.index[::-1], miss.values[::-1], color="#4575b4")
ax.set_xlabel("% missing"); ax.set_title("Missing values by column")
save(fig, "01_missing_values")

# 2. Score distribution (class imbalance)
print("\nScore distribution:")
vc = lab.Score.value_counts().sort_index()
print((vc / vc.sum()).round(4).to_string())
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(vc.index, vc.values, color=SCORE_COLORS)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v / vc.sum():.1%}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Score"); ax.set_ylabel("Reviews"); ax.set_title("Score distribution (class imbalance)")
save(fig, "02_score_distribution")

# 3. Engineered numeric features
GOOD = ["masterpiece", "brilliant", "perfect", "classic", "amazing", "incredible", "essential",
        "stunning", "flawless", "gem", "favorite", "best", "excellent", "great", "love",
        "beautiful", "genius", "superb"]
BAD = ["boring", "disappointing", "disappointed", "awful", "terrible", "worst", "bland",
       "mediocre", "forgettable", "waste", "weak", "dull", "meh", "generic", "filler", "mess",
       "unlistenable", "overrated", "bad", "poor"]
good_re = re.compile(r"\b(?:" + "|".join(GOOD) + r")\b")
bad_re = re.compile(r"\b(?:" + "|".join(BAD) + r")\b")
neg_re = re.compile(r"\b(?:not|no|never|without|hardly)\b|n't")


def build_features(d, ref):
    f = pd.DataFrame(index=d.index)
    rt = d.reviewText.fillna(""); sm = d.summary.fillna("")
    low = (sm + " . " + rt).str.lower()
    f["VotedHelpful"] = d.VotedHelpful.fillna(0)
    f["TotalVotes"] = d.TotalVotes.fillna(0)
    f["helpful_ratio"] = f.VotedHelpful / (f.TotalVotes + 1)
    f["text_len"] = rt.str.len()
    f["summary_len"] = sm.str.len()
    f["n_words"] = rt.str.split().str.len().fillna(0)
    f["n_genres"] = d.genres.fillna("").apply(lambda s: len([g for g in s.split(",") if g.strip()]))
    f["exclaims"] = rt.str.count("!")
    f["questions"] = rt.str.count(r"\?")
    f["caps_ratio"] = rt.str.count(r"[A-Z]") / (f.text_len + 1)
    f["negations"] = low.str.count(neg_re)
    f["seed_good"] = low.str.count(good_re)
    f["seed_bad"] = low.str.count(bad_re)
    f["seed_net"] = f.seed_good - f.seed_bad
    f["year"] = pd.to_datetime(d.unixReviewTime, unit="s", errors="coerce").dt.year
    for c, nm in [("reviewerID", "reviewer_n"), ("album_mbid", "album_n"), ("artist_mbid", "artist_n")]:
        f[nm] = d[c].map(ref[c].value_counts())     # counts over ALL rows (no labels used)
    return f

feats = build_features(lab, df)
feats["Score"] = lab.Score

# 4. Mean score vs. each numeric feature (binned)
print("\nScore vs numeric features...")
panels = ["text_len", "summary_len", "n_words", "TotalVotes", "helpful_ratio",
          "exclaims", "seed_net", "reviewer_n", "album_n"]
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, col in zip(axes.ravel(), panels):
    x = feats[col]
    try:
        bins = pd.qcut(x, q=10, duplicates="drop")
    except ValueError:
        bins = pd.cut(x, bins=10)
    g = feats.groupby(bins, observed=True).Score.mean()
    ax.plot(range(len(g)), g.values, marker="o", color="#4575b4")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{i.mid:.3g}" for i in g.index], rotation=45, fontsize=7)
    ax.axhline(feats.Score.mean(), ls="--", color="gray", lw=0.8)
    ax.set_title(f"Mean score by {col} (deciles)", fontsize=9)
    ax.set_ylabel("mean score")
save(fig, "03_score_vs_features")


# 5. Distributions split by score
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, cap in zip(axes, ["text_len", "summary_len", "exclaims"], [3000, 80, 10]):
    data = [feats.loc[feats.Score == s, col].clip(upper=cap) for s in range(1, 6)]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([1, 2, 3, 4, 5])
    for patch, c in zip(bp["boxes"], SCORE_COLORS):
        patch.set_facecolor(c)
    ax.set_title(f"{col} by score"); ax.set_xlabel("Score")
save(fig, "04_boxplots_by_score")

# 6. Time trends
yr = feats.dropna(subset=["year"]).groupby("year").Score.agg(["mean", "count"])
yr = yr[yr["count"] >= 30]
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.bar(yr.index, yr["count"], color="#bbbbbb")
ax1.set_ylabel("reviews"); ax1.set_xlabel("year")
ax2 = ax1.twinx()
ax2.plot(yr.index, yr["mean"], color="#d73027", marker="o")
ax2.set_ylabel("mean score", color="#d73027")
ax1.set_title("Reviews and mean score over time")
save(fig, "05_time_trend")


# 7. Genres
gen = lab[["genres", "Score"]].dropna().copy()
gen["genre"] = gen.genres.str.split(",")
gen = gen.explode("genre"); gen["genre"] = gen.genre.str.strip()
gs = gen.groupby("genre").Score.agg(["mean", "count"])
gs = gs[gs["count"] >= 200].sort_values("count", ascending=False).head(25)
print(gs.round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(gs.index[::-1], gs["count"][::-1], color="#4575b4")
axes[0].set_title("Most common genres")
gm = gs.sort_values("mean")
axes[1].barh(gm.index, gm["mean"], color="#91cf60")
axes[1].axvline(lab.Score.mean(), ls="--", color="gray")
axes[1].set_xlim(max(1, gm["mean"].min() - 0.2), gm["mean"].max() + 0.1)
axes[1].set_title("Mean score by genre")
save(fig, "06_genres")

# =====================================================================
# 8. Reviewer / album / artist behaviour
# =====================================================================
print("\nReviewer behaviour...")
rv = lab.groupby("reviewerID").Score.agg(["mean", "std", "count"])
print(f"reviewers={len(rv):,}  median reviews/reviewer={rv['count'].median():.0f}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(np.log10(rv["count"]), bins=40, color="#4575b4")
axes[0].set_xlabel("log10(reviews per reviewer)"); axes[0].set_title("Reviewer activity")
heavy = rv[rv["count"] >= 5]
axes[1].hist(heavy["mean"], bins=40, color="#91cf60")
axes[1].set_xlabel("reviewer mean score (>=5 reviews)"); axes[1].set_title("Reviewers are consistent raters")
axes[2].hist(heavy["std"].dropna(), bins=40, color="#fc8d59")
axes[2].set_xlabel("reviewer score std (>=5 reviews)"); axes[2].set_title("Within-reviewer spread")
save(fig, "07_reviewer_behaviour")

# Does target encoding work? How many unlabeled rows share a key with labeled data?
print("\nKey overlap (unlabeled rows whose key appears in labeled data):")
for c in ["reviewerID", "album_mbid", "artist_mbid"]:
    seen = unl[c].isin(set(lab[c].dropna())).mean()
    print(f"  {c:12s}: {seen:.1%}")

# 9. Good / bad words (data-driven polarity)
print("\nGood/bad words...")
txt = (lab.summary.fillna("") + " . " + lab.reviewText.fillna(""))
cv = CountVectorizer(ngram_range=(1, 2), min_df=30, max_features=150_000,
                     binary=True, strip_accents="unicode", dtype=np.float32)
B = cv.fit_transform(txt)
terms = cv.get_feature_names_out()
yy = lab.Score.values.astype(float); g = yy.mean(); K = 20
cnt = np.asarray(B.sum(0)).ravel()
mean_sc = (np.asarray(B.T.dot(yy)).ravel() + K * g) / (cnt + K)
wp = pd.DataFrame({"term": terms, "mean_score": mean_sc, "n": cnt.astype(int)})
wp["polarity"] = wp.mean_score - g
wp["impact"] = wp.polarity.abs() * np.sqrt(wp.n)
wp.sort_values("polarity").to_csv(f"{OUT}/word_polarity.csv", index=False)

top = 20
bad_w = wp[wp.n >= 100].sort_values("polarity").head(top)
good_w = wp[wp.n >= 100].sort_values("polarity").tail(top)
print("\nWorst terms:\n", bad_w[["term", "mean_score", "n"]].to_string(index=False))
print("\nBest terms:\n", good_w.iloc[::-1][["term", "mean_score", "n"]].to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].barh(bad_w.term[::-1], bad_w.mean_score[::-1], color="#d73027")
axes[0].set_title(f"Lowest-scoring terms (n>=100)"); axes[0].axvline(g, ls="--", color="gray")
axes[1].barh(good_w.term, good_w.mean_score, color="#1a9850")
axes[1].set_title(f"Highest-scoring terms (n>=100)"); axes[1].axvline(g, ls="--", color="gray")
for ax in axes:
    ax.set_xlabel("mean score of reviews containing term")
save(fig, "08_good_bad_words")

fig, ax = plt.subplots(figsize=(7, 5))
samp = wp.sample(min(len(wp), 20000), random_state=0)
ax.scatter(np.log10(samp.n), samp.mean_score, s=4, alpha=0.3, color="#4575b4")
ax.axhline(g, ls="--", color="gray")
ax.set_xlabel("log10(term frequency)"); ax.set_ylabel("mean score")
ax.set_title("Term polarity vs frequency (rare terms are noisy -> smoothing)")
save(fig, "09_polarity_vs_frequency")

# Seed-word check
low = txt.str.lower()
rows = []
for name, rx in [("good seed words", good_re), ("bad seed words", bad_re), ("negations", neg_re)]:
    has = low.str.contains(rx).values
    rows.append((name, yy[has].mean(), yy[~has].mean(), has.mean()))
print("\nSeed words (mean score present vs absent):")
print(pd.DataFrame(rows, columns=["list", "present", "absent", "share_of_reviews"]).round(3).to_string(index=False))


# 10. Correlation matrix
print("\nCorrelation matrix...")
num_cols = ["Score", "VotedHelpful", "TotalVotes", "helpful_ratio", "text_len", "summary_len",
            "n_words", "n_genres", "exclaims", "questions", "caps_ratio", "negations",
            "seed_good", "seed_bad", "seed_net", "year", "reviewer_n", "album_n", "artist_n"]
corr = feats[num_cols].corr(method="spearman")
print("\nSpearman correlation with Score:\n", corr.Score.drop("Score").sort_values().round(3).to_string())

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=8)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Spearman correlation matrix")
save(fig, "10_correlation_matrix")

# Correlation with Score as a ranked bar chart
cs = corr.Score.drop("Score").sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(cs.index, cs.values, color=["#d73027" if v < 0 else "#1a9850" for v in cs.values])
ax.set_xlabel("Spearman correlation with Score"); ax.set_title("Which features relate to Score?")
save(fig, "11_correlation_with_score")

# Strongly collinear pairs (useful for pruning features)
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pairs = pairs[pairs.abs() > 0.7].sort_values(key=abs, ascending=False)
print("\nHighly correlated feature pairs (|rho|>0.7):")
print(pairs.round(3).to_string() if len(pairs) else "  none")

print(f"\nDone. Figures and word_polarity.csv are in ./{OUT}/")