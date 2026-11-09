import os
import time
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# Fixed seed for repeatable cross-validation splits and model training.
SEED = 42

# Number of folds used to create predictions for the second-stage models.
K = 5

# Reuse text-model predictions across runs; delete this cache if models change
CACHE = "cache_text_v3.npz"

# Threshold for 100% submission score
TARGET = 0.5539

start = time.time()

# Load datasets
df = pd.read_csv("data/train.csv")
sample = pd.read_csv("data/sample_submission.csv") #Used to format the submission file

train = df[df["Score"].notna()].copy().reset_index(drop=True)
test = df[df["Score"].isna()].copy().reset_index(drop=True)
y = train["Score"].astype(int).to_numpy()
print(f"Labeled: {len(train)} | Unlabeled: {len(test)}", flush=True)

genres = ["rock", "pop", "jazz", "metal", "electronic",
          "hip hop", "folk", "country", "classical", "blues"]
keys = ["reviewerID", "artist_mbid", "album_mbid", "rev_art"]

# Feature engineeing: convert text to features; add numeric features
def add_features(data):
    data = data.copy()

    # Missing reviews becomes empty strings
    summary = data["summary"].fillna("").astype(str)
    review = data["reviewText"].fillna("").astype(str)

    data["summary_text"] = summary
    data["review_text"] = review
    data["year"] = pd.to_datetime(
        data["unixReviewTime"], unit="s", errors="coerce").dt.year
    data["length"] = np.log1p(review.str.len())
    data["exclamations"] = review.str.count("!")
    data["capital_words"] = review.str.count(r"[A-Z]{3,}")
    data["summary_length"] = np.log1p(summary.str.len())
    data["summary_words"] = np.log1p(summary.str.split().str.len())
    data["summary_exclamations"] = summary.str.count("!")
    data["summary_questions"] = summary.str.count(r"\?")

    #  Add one to the denominator so reviews with zero votes are smooth
    votes = pd.to_numeric(data["TotalVotes"], errors="coerce").fillna(0).clip(lower=0)
    helpful = pd.to_numeric(data["VotedHelpful"], errors="coerce").fillna(0).clip(lower=0)
    data["helpful_ratio"] = helpful / (votes + 1)
    data["log_votes"] = np.log1p(votes)

    # Add genre features
    genre_text = data["genres"].fillna("").astype(str).str.lower()
    data["genre_count"] = genre_text.apply(
        lambda x: len([g for g in x.split(",") if g.strip()]))
    for g in genres:
        data["genre_" + g.replace(" ", "_")] = (
            genre_text.str.contains(g, regex=False).astype(int))

    # Combination of reviewer and artist for cross-features
    data["rev_art"] = (data["reviewerID"].astype(str) + "_"
                       + data["artist_mbid"].astype(str))
    return data


train = add_features(train)
test = add_features(test)
all_data = pd.concat([train, test], ignore_index=True)

# Combine train and test for counts/avgs
for col in keys:
    counts = all_data[col].value_counts()
    train[col + "_count"] = train[col].map(counts).fillna(0)
    test[col + "_count"] = test[col].map(counts).fillna(0)

for key, prefix in [("reviewerID", "reviewer"), ("artist_mbid", "artist")]:
    for col in ["length", "exclamations", "helpful_ratio", "log_votes"]:
        averages = all_data.groupby(key)[col].mean()
        train[prefix + "_avg_" + col] = train[key].map(averages)
        test[prefix + "_avg_" + col] = test[key].map(averages)

for d in (train, test):
    d["length_vs_reviewer"] = d["length"] - d["reviewer_avg_length"]


# Count how many reviews each reviewer wrote on each day
day_counts = all_data.groupby(["reviewerID", "unixReviewTime"]).size()
for d in (train, test):
    d["reviewer_day_count"] = (
        d.set_index(["reviewerID", "unixReviewTime"])
        .index.map(day_counts).astype(float))

# Compare each review year with the artist’s usual review year
artist_year = all_data.groupby("artist_mbid")["year"].median()
for d in (train, test):
    d["year_vs_artist"] = d["year"] - d["artist_mbid"].map(artist_year)

cols = [
    "year", "length", "exclamations", "capital_words", "helpful_ratio",
    "log_votes", "genre_count", "summary_length", "summary_words",
    "summary_exclamations", "summary_questions", "length_vs_reviewer",
    "reviewer_day_count", "year_vs_artist",
]
cols += [k + "_count" for k in keys]
for prefix in ["reviewer", "artist"]:
    cols += [prefix + "_avg_" + c for c in
             ["length", "exclamations", "helpful_ratio", "log_votes"]]
cols += ["genre_" + g.replace(" ", "_") for g in genres]

# Text models: Logistic Regression, Complement Naive Bayes, Ridge Regression
# Use the same five stratified folds for text and rating-history encodings
# This ensures that the second-stage models are trained on out-of-fold predictions from the first-stage models.

splits = list(StratifiedKFold(K, shuffle=True, random_state=SEED)
              .split(np.zeros(len(train)), y))

# Loads cached text-model predictions if they exist, otherwise creates them from scratch
if os.path.exists(CACHE):
    print("Loading cached text predictions...", flush=True)
    z = np.load(CACHE)
    oof_lr, test_lr = z["oof_lr"], z["test_lr"]
    oof_nb, test_nb = z["oof_nb"], z["test_nb"]
    oof_rg, test_rg = z["oof_rg"], z["test_rg"]
else:
    print("Creating TF-IDF features...", flush=True)
    summary_tf = TfidfVectorizer(min_df=3, max_features=20000, ngram_range=(1, 2),
                                 sublinear_tf=True, dtype=np.float32)
    review_tf = TfidfVectorizer(min_df=3, max_features=50000, ngram_range=(1, 2),
                                sublinear_tf=True, dtype=np.float32)
    sum_char_tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                                  max_features=60000, sublinear_tf=True,
                                  dtype=np.float32)
    # TF-IDF learns vocab from text only, without using Score label
    summary_tf.fit(all_data["summary_text"])
    review_tf.fit(all_data["review_text"])
    sum_char_tf.fit(all_data["summary_text"])
    print(f"TF-IDF fitted ({(time.time()-start)/60:.1f} min)", flush=True)

    # Summary words are weighted more heavily 
    def build(d):
        return hstack([summary_tf.transform(d["summary_text"]) * 2,
                       review_tf.transform(d["review_text"]),
                       sum_char_tf.transform(d["summary_text"])], format="csr")

    X_text = build(train)
    X_test_text = build(test)

    oof_lr = np.zeros((len(train), 5)); test_lr = np.zeros((len(test), 5))
    oof_nb = np.zeros((len(train), 5)); test_nb = np.zeros((len(test), 5))
    oof_rg = np.zeros(len(train));      test_rg = np.zeros(len(test))
    for fold, (a, b) in enumerate(splits, 1):
        print(f"Text models fold {fold}/{K}  ({(time.time()-start)/60:.1f} min)", flush=True)
        m = LogisticRegression(C=2, max_iter=100)
        m.fit(X_text[a], y[a])
        oof_lr[b] = m.predict_proba(X_text[b])
        test_lr += m.predict_proba(X_test_text) / K

        nb = ComplementNB(alpha=0.3)
        nb.fit(X_text[a], y[a])
        oof_nb[b] = nb.predict_proba(X_text[b])
        test_nb += nb.predict_proba(X_test_text) / K

        rg = Ridge(alpha=3.0, solver="sparse_cg")
        rg.fit(X_text[a], y[a])
        oof_rg[b] = rg.predict(X_text[b])
        test_rg += rg.predict(X_test_text) / K
    np.savez(CACHE, oof_lr=oof_lr, test_lr=test_lr, oof_nb=oof_nb, test_nb=test_nb,
             oof_rg=oof_rg, test_rg=test_rg)
    del X_text, X_test_text

print("LR  text OOF macro F1:", round(f1_score(y, oof_lr.argmax(1) + 1, average="macro"), 4))
print("NB  text OOF macro F1:", round(f1_score(y, oof_nb.argmax(1) + 1, average="macro"), 4))
print("Ridge text OOF RMSE  :", round(float(np.sqrt(np.mean((oof_rg - y) ** 2))), 4), flush=True)


# Rating-history encoding
def target_encoding(source, target, col):
    mean = source["Score"].mean()
    high_mean = (source["Score"] == 5).mean()
    low_mean = (source["Score"] <= 2).mean()
    sq_mean = (source["Score"] ** 2).mean()

    s = source[[col, "Score"]].copy()
    s["high"] = (s["Score"] == 5).astype(int)
    s["low"] = (s["Score"] <= 2).astype(int)
    s["sq"] = s["Score"] ** 2
    stats = s.groupby(col).agg(total=("Score", "sum"), count=("Score", "size"),
                               high=("high", "sum"), low=("low", "sum"),
                               sq=("sq", "sum"))

    count = target[col].map(stats["count"]).fillna(0).to_numpy()
    total = target[col].map(stats["total"]).fillna(0).to_numpy()
    high = target[col].map(stats["high"]).fillna(0).to_numpy()
    low = target[col].map(stats["low"]).fillna(0).to_numpy()
    sq = target[col].map(stats["sq"]).fillna(0).to_numpy()

    te = (total + 10 * mean) / (count + 10)
    # Estimate rating variance 
    var = (sq + 10 * sq_mean) / (count + 10) - te ** 2
    return pd.DataFrame({
        col + "_te": te,
        col + "_te_count": count,
        col + "_high": (high + 10 * high_mean) / (count + 10),
        col + "_low": (low + 10 * low_mean) / (count + 10),
        col + "_std": np.sqrt(np.clip(var, 0, None)),
    })

# Build features for the second stage models
print("Building stacker features...", flush=True)
X_train = train[cols].reset_index(drop=True).copy()
X_test = test[cols].reset_index(drop=True).copy()
for i in range(5):
    X_train[f"p{i+1}"] = oof_lr[:, i]
    X_test[f"p{i+1}"] = test_lr[:, i]
    X_train[f"nb{i+1}"] = oof_nb[:, i]
    X_test[f"nb{i+1}"] = test_nb[:, i]

scores = np.arange(1, 6)
X_train["text_exp"] = oof_lr @ scores
X_test["text_exp"] = test_lr @ scores
X_train["ridge"] = oof_rg
X_test["ridge"] = test_rg

# Estimate reviewer / album average text expectations (how positive their text usually is)
tmp = pd.DataFrame({
    "rev": pd.concat([train["reviewerID"], test["reviewerID"]], ignore_index=True),
    "alb": pd.concat([train["album_mbid"], test["album_mbid"]], ignore_index=True),
    "exp": np.r_[X_train["text_exp"].to_numpy(), X_test["text_exp"].to_numpy()],
    "rg": np.r_[oof_rg, test_rg],
})
n_tr = len(train)
for k, name in [("rev", "reviewer"), ("alb", "album")]:
    for v, suffix in [("exp", "_avg_text_exp"), ("rg", "_avg_ridge")]:
        m = tmp.groupby(k)[v].mean()
        X_train[name + suffix] = tmp[k].iloc[:n_tr].map(m).to_numpy()
        X_test[name + suffix] = tmp[k].iloc[n_tr:].map(m).to_numpy()

for col in keys:
    # Create out-of-fold target encoding for each key column
    names = [col + s for s in ["_te", "_te_count", "_high", "_low", "_std"]]
    oof_te = pd.DataFrame(0.0, index=range(len(train)), columns=names)
    for a, b in splits:
        enc = target_encoding(train.iloc[a], train.iloc[b], col)
        oof_te.iloc[b] = enc[names].to_numpy()
    for n in names:
        X_train[n] = oof_te[n].to_numpy()

    test_te = target_encoding(train, test, col)
    for n in names:
        X_test[n] = test_te[n].to_numpy()
    X_test[col + "_te_count"] *= (K - 1) / K   # match OOF scale

# Reviewer bias compares historical ratings 
for X in (X_train, X_test):
    X["reviewer_bias"] = X["reviewerID_te"] - X["reviewer_avg_text_exp"]
    X["reviewer_bias_ridge"] = X["reviewerID_te"] - X["reviewer_avg_ridge"]
    X["ridge_vs_reviewer"] = X["ridge"] - X["reviewer_avg_ridge"]

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
medians = X_train.median().fillna(0)
# Replace missing values with median of training set, then fill any remaining NaNs with 0
X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)
X_test = X_test[X_train.columns]
print("Stacker features:", X_train.shape[1], flush=True)

# Implement random forest
def make_rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
        random_state=SEED, class_weight="balanced_subsample")

# Implement extra trees
def make_et():
    return ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
        random_state=SEED, class_weight="balanced_subsample")

# Compare models
print("Tree-model OOF...", flush=True)
oof_rf = np.zeros((len(train), 5))
oof_et = np.zeros((len(train), 5))
for fold, (a, b) in enumerate(
        StratifiedKFold(K, shuffle=True, random_state=SEED + 1).split(X_train, y), 1):
    print(f"Fold {fold}/{K}  ({(time.time()-start)/60:.1f} min)", flush=True)
    rf = make_rf(); rf.fit(X_train.iloc[a], y[a])
    oof_rf[b] = rf.predict_proba(X_train.iloc[b])
    et = make_et(); et.fit(X_train.iloc[a], y[a])
    oof_et[b] = et.predict_proba(X_train.iloc[b])

# Evaluate and tune weights for each model
def tune_weights(probs, labels):
    w = np.ones(5)
    best = f1_score(labels, probs.argmax(1) + 1, average="macro")
    for _ in range(2):
        for i in range(5):
            for v in np.arange(0.5, 2.51, 0.1):
                trial = w.copy(); trial[i] = v
                s = f1_score(labels, (probs * trial).argmax(1) + 1, average="macro")
                if s > best:
                    best, w = s, trial
    return w, best

# Evaluate blend
candidates = {
    "RF": oof_rf,
    "ET": oof_et,
    "Blend 50/50": 0.5 * oof_rf + 0.5 * oof_et,
}
results = {}
for name, probs in candidates.items():
    raw = f1_score(y, probs.argmax(1) + 1, average="macro")
    w, tuned = tune_weights(probs, y)
    results[name] = (tuned, w, probs)
    print(f"{name:12s} OOF macro F1 raw {raw:.4f} | tuned {tuned:.4f} | weights {np.round(w, 2)}",
          flush=True)

best_name = max(results, key=lambda k: results[k][0])
tuned, weights, best_oof = results[best_name]
print(f"\nBest: {best_name}  tuned OOF macro F1 = {tuned:.4f}")
print(classification_report(y, (best_oof * weights).argmax(1) + 1, digits=4))
if tuned < TARGET:
    print(f"OOF {tuned:.4f} is below your {TARGET} target; writing submission.csv "
          "anyway, but don't submit it yet.")
# Refit each tree ensenble on full training set to create final predictions
print("Fitting final models...", flush=True)
final_rf = make_rf(); final_rf.fit(X_train, y)
final_et = make_et(); final_et.fit(X_train, y)
p_rf = final_rf.predict_proba(X_test)
p_et = final_et.predict_proba(X_test)
final_probs = {"RF": p_rf, "ET": p_et, "Blend 50/50": 0.5 * p_rf + 0.5 * p_et}[best_name]
predictions = final_rf.classes_[np.argmax(final_probs * weights, axis=1)].astype(int)

# Match predictions
lookup = pd.Series(predictions, index=test["id"].to_numpy())
submission = sample.copy()
submission["Score"] = submission["id"].map(lookup)
assert submission["Score"].notna().all()
assert submission["Score"].isin([1, 2, 3, 4, 5]).all()
submission["Score"] = submission["Score"].astype(int)
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv")
print(submission["Score"].value_counts().sort_index())
print(f"Total runtime: {(time.time() - start) / 60:.1f} minutes")