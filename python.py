import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning)
start_time = time.time()


# Settings
use_ridge = True              # Add a numerical rating estimate
fold_jobs = 3                 # Limit parallel jobs to manage memory
lexicon_smoothing = 20        # Reduce the influence of rare words
lexicon_threshold = 0.30      # Minimum strength for positive/negative words
n_genres = 25                 # Use the most common genres
ordinal_c = 1.0               # Regularization for ordinal LS
rf_trees, rf_min_leaf, rf_max_features = 400, 5, "sqrt"
use_extra_trees = True        # Test whether blending forests improves F1
et_trees, et_min_leaf, et_max_features = 400, 3, 0.3

# Cache expensive models to avoid retraining every run
stage1_cache = f"cache/stage1_v3_r{int(use_ridge)}_k{lexicon_smoothing}_t{lexicon_threshold}.npz"
ordinal_cache = f"cache/ordinal_v1_C{ordinal_c}.npz"
os.makedirs("cache", exist_ok=True)


# Load data
print("Loading datasets...")
data = pd.read_csv("data/train.csv")
train = data[data.Score.notna()].reset_index(drop=True)   # reviews with a score
test = data[data.Score.isna()].reset_index(drop=True)     # reviews to predict
y = train.Score.astype(int).values
print(f"  train={len(train)}  test={len(test)}")

# Verify matches
if os.path.exists("data/test.csv"):
    expected_ids = set(pd.read_csv("data/test.csv").iloc[:, 0])
    print("  test.csv ids match the reviews to predict:", expected_ids == set(test.id))

# Use consistent folds
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(train, y))


def macro_f1(y_true, y_pred, n_classes=5):
    # Calc macro F1 during tuning
    confusion = np.bincount((y_true - 1) * n_classes + (y_pred - 1),
                            minlength=n_classes ** 2).reshape(n_classes, n_classes)
    true_pos = np.diag(confusion).astype(float)
    precision = true_pos / np.maximum(confusion.sum(0), 1)
    recall = true_pos / np.maximum(confusion.sum(1), 1)
    return (2 * precision * recall / np.maximum(precision + recall, 1e-12)).mean()



# Turn review text and summaries into TF-IDF
tfidf_store = {}

# Build and reuse features
def get_tfidf_matrices():
    if not tfidf_store:
        print("Vectorizing text...")
        review_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000,
                                       sublinear_tf=True, strip_accents="unicode")
        summary_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                                        sublinear_tf=True, strip_accents="unicode")
        tfidf_store["train"] = hstack([review_tfidf.fit_transform(train.reviewText.fillna("")),
                                       summary_tfidf.fit_transform(train.summary.fillna(""))],
                                      format="csr")
        tfidf_store["test"] = hstack([review_tfidf.transform(test.reviewText.fillna("")),
                                      summary_tfidf.transform(test.summary.fillna(""))],
                                     format="csr")
    return tfidf_store["train"], tfidf_store["test"]

# # Learn which words are associated with higher or lower ratings
def word_polarity(present, scores):
    # present is a reviews by words matrix of 0 and 1; returns one polarity per word
    overall_mean = scores.mean()
    reviews_with_word = np.asarray(present.sum(0)).ravel()
    score_total = np.asarray(present.T.dot(scores.astype(float))).ravel()
    return ((score_total + lexicon_smoothing * overall_mean)
            / (reviews_with_word + lexicon_smoothing) - overall_mean)

lexicon_columns = ["lex_mean", "lex_max", "lex_min", "lex_good", "lex_bad",
                   "lex_good_frac", "lex_bad_frac", "lex_net", "lex_strong_good", "lex_strong_bad"]

def lexicon_features(present, polarity):
    n_words = np.maximum(present.getnnz(axis=1), 1).astype(float)
    weighted = present.multiply(polarity[None, :]).tocsr()

    def count_words(mask):
        return np.asarray(present.dot(mask.astype(float))).ravel()

    good = count_words(polarity > lexicon_threshold)
    bad = count_words(polarity < -lexicon_threshold)
    return np.column_stack([
        np.asarray(weighted.sum(1)).ravel() / n_words,    # average polarity
        weighted.max(axis=1).toarray().ravel(),           # most positive word
        weighted.min(axis=1).toarray().ravel(),           # most negative word
        good, bad, good / n_words, bad / n_words, good - bad,
        count_words(polarity > 2 * lexicon_threshold),    # strongly good words
        count_words(polarity < -2 * lexicon_threshold),   # strongly bad words
    ])


# Train text-based models
if os.path.exists(stage1_cache):
    print(f"Loading cached stage 1 results from {stage1_cache}")
    cached = np.load(stage1_cache)
    oof_lr, test_lr = cached["oof_lr"], cached["te_lr"]
    oof_ridge, test_ridge = cached["oof_rd"], cached["te_rd"]
    oof_lexicon, test_lexicon = cached["lex_l"], cached["lex_u"]
else:
    x_train_text, x_test_text = get_tfidf_matrices()

    # Use word presence
    present_train = x_train_text.copy(); present_train.data[:] = 1.0
    present_test = x_test_text.copy(); present_test.data[:] = 1.0

    print("Building out of fold lexicon features...")
    oof_lexicon = np.zeros((len(train), len(lexicon_columns)))
    for fit_idx, val_idx in folds:
        polarity = word_polarity(present_train[fit_idx], y[fit_idx])
        oof_lexicon[val_idx] = lexicon_features(present_train[val_idx], polarity)
    test_lexicon = lexicon_features(present_test, word_polarity(present_train, y))
    del present_train, present_test

    print("Training Logistic Regression...")
    log_reg = LogisticRegression(C=1.0, max_iter=400, tol=1e-3, class_weight="balanced")
    oof_lr = cross_val_predict(log_reg, x_train_text, y, cv=folds, method="predict_proba", n_jobs=fold_jobs)
    test_lr = log_reg.fit(x_train_text, y).predict_proba(x_test_text)

    # Logistic regression estimates the probability of each rating
    if use_ridge:
        print("Training Ridge...")
        ridge = Ridge(alpha=2.0)
        oof_ridge = cross_val_predict(ridge, x_train_text, y, cv=folds, n_jobs=fold_jobs)
        test_ridge = ridge.fit(x_train_text, y).predict(x_test_text)
    else:
        oof_ridge, test_ridge = np.zeros(len(train)), np.zeros(len(test))

    np.savez(stage1_cache, oof_lr=oof_lr, te_lr=test_lr, oof_rd=oof_ridge, te_rd=test_ridge,
             lex_l=oof_lexicon, lex_u=test_lexicon)
    print(f"  saved to {stage1_cache}")



# Model the ordered relationship between ratings; predicts whether a review scores at least 2, 3, 4, or 5
thresholds = [2, 3, 4, 5]
if os.path.exists(ordinal_cache):
    print(f"Loading cached ordinal results from {ordinal_cache}")
    cached = np.load(ordinal_cache)
    oof_ordinal, test_ordinal = cached["oof_ord"], cached["te_ord"]
else:
    x_train_text, x_test_text = get_tfidf_matrices()
    oof_ordinal = np.zeros((len(train), len(thresholds)))
    test_ordinal = np.zeros((len(test), len(thresholds)))
    for j, threshold in enumerate(thresholds):
        print(f"Training ordinal LR: score >= {threshold}...")
        at_least = (y >= threshold).astype(int)
        ordinal_lr = LogisticRegression(C=ordinal_c, max_iter=400, tol=1e-3)
        oof_ordinal[:, j] = cross_val_predict(ordinal_lr, x_train_text, at_least, cv=folds,
                                              method="predict_proba", n_jobs=fold_jobs)[:, 1]
        test_ordinal[:, j] = ordinal_lr.fit(x_train_text, at_least).predict_proba(x_test_text)[:, 1]
    np.savez(ordinal_cache, oof_ord=oof_ordinal, te_ord=test_ordinal)
    print(f"  saved to {ordinal_cache}")
tfidf_store.clear()


# Metadata features for forest models
print("Building metadata features...")
all_rows = pd.concat([train, test], ignore_index=True)    # Used only for counts that need no labels
entity_columns = {"reviewer": "reviewerID", "album": "album_mbid", "artist": "artist_mbid"}
review_counts = {name: all_rows[col].map(all_rows[col].value_counts())
                 for name, col in entity_columns.items()}

top_genres = (all_rows.genres.dropna().str.split(",").explode().str.strip()
              .value_counts().head(n_genres).index.tolist())

# Common word tracking
good_words = ["masterpiece", "brilliant", "perfect", "classic", "amazing", "incredible", "essential",
              "stunning", "flawless", "gem", "favorite", "best", "excellent", "great", "love",
              "beautiful", "genius", "superb"]
bad_words = ["boring", "disappointing", "disappointed", "awful", "terrible", "worst", "bland",
             "mediocre", "forgettable", "waste", "weak", "dull", "meh", "generic", "filler", "mess",
             "unlistenable", "overrated", "bad", "poor"]
good_word_pattern = re.compile(r"\b(?:" + "|".join(good_words) + r")\b")
bad_word_pattern = re.compile(r"\b(?:" + "|".join(bad_words) + r")\b")
negation_pattern = re.compile(r"\b(?:not|no|never|without|hardly)\b|n't")

# Ratings written in the review text ("4/5" or "two stars")
number_words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
fraction_pattern = r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*(?:/|out of)\s*(5|10)\b"
stars_pattern = r"\b(zero|one|two|three|four|five|[0-5](?:\.5)?)[\s-]*stars?\b"


def written_rating(text):
    """Extract explicit ratings and convert them to a 0–5 scale"""
    fraction = text.str.extract(fraction_pattern)
    numerator = pd.to_numeric(fraction[0], errors="coerce")
    denominator = pd.to_numeric(fraction[1], errors="coerce")
    from_fraction = (numerator / denominator * 5).where(numerator <= denominator)
    stars = text.str.extract(stars_pattern)[0]
    from_stars = stars.map(number_words).fillna(pd.to_numeric(stars, errors="coerce"))
    rating = from_fraction.fillna(from_stars).astype(float)
    return rating.fillna(-1), rating.notna().astype(int)


def basic_features(reviews, offset):
    """Create features for review quality, writing style, and metadata"""
    f = pd.DataFrame(index=reviews.index)
    review_text, summary = reviews.reviewText.fillna(""), reviews.summary.fillna("")
    text = (summary + " . " + review_text).str.lower()
    total_votes = reviews.TotalVotes.fillna(0)
    helpful_votes = reviews.VotedHelpful.fillna(0)

    # Votes and time
    f["total_votes"] = total_votes
    f["unhelpful_votes"] = (total_votes - helpful_votes).clip(lower=0)
    f["helpful_ratio"] = helpful_votes / (total_votes + 1)
    f["time"] = reviews.unixReviewTime.fillna(reviews.unixReviewTime.median())

    # Text style
    f["text_len"] = review_text.str.len()
    f["summary_len"] = summary.str.len()
    f["exclamations"] = review_text.str.count("!")
    f["questions"] = review_text.str.count(r"\?")
    f["caps_ratio"] = review_text.str.count(r"[A-Z]") / (f["text_len"] + 1)

    # Sentiment words
    f["negations"] = text.str.count(negation_pattern)
    f["bad_words"] = text.str.count(bad_word_pattern)
    f["net_good_words"] = text.str.count(good_word_pattern) - f.bad_words
    f["summary_bad_words"] = summary.str.lower().str.count(bad_word_pattern)
    f["written_rating"], f["has_written_rating"] = written_rating(text)

    # Genre indicators account for differences between music categorie
    genre_sets = reviews.genres.fillna("").apply(
        lambda s: {g.strip() for g in s.split(",") if g.strip()})
    f["n_genres"] = genre_sets.apply(len)
    for genre in top_genres:
        column = "genre_" + re.sub(r"\W+", "_", genre.lower()).strip("_")
        f[column] = genre_sets.apply(lambda s, genre=genre: int(genre in s))

    # Review counts measure how frequently an entity appears
    positions = np.arange(len(reviews)) + offset
    for name in entity_columns:
        f[f"{name}_total_reviews"] = review_counts[name].iloc[positions].values
    return f


x_train = basic_features(train, 0)
x_test = basic_features(test, len(train))



# Calculate reviewer, album, and artist rating tendencies
# Out-of-fold statistics help prevent the model from memorizing labels
print("Calculating out of fold key statistics...")
score_values = np.arange(1, 6)
text_score = oof_ridge if use_ridge else oof_lr @ score_values        # Text only prediction
text_score_test = test_ridge if use_ridge else test_lr @ score_values
residual = y - text_score

score_onehot = np.eye(5)[y - 1]
global_mean = y.mean()
global_mean_square = (y.astype(float) ** 2).mean()
global_shares = score_onehot.mean(0)
entity_smoothing = 5   


def entity_totals(entity_ids, rows):
    """For each entity: sums of score, squared score, residual and per score counts over `rows`."""
    d = pd.DataFrame(score_onehot[rows], columns=[f"c{i}" for i in range(5)])
    d["y"], d["y2"], d["r"] = y[rows], y[rows].astype(float) ** 2, residual[rows]
    d["n"], d["entity"] = 1.0, entity_ids
    return d.groupby("entity").sum()


def entity_features(totals, entity_ids, prefix, with_score_shares, count_scale=1.0):
    t = totals.reindex(entity_ids)              # Never seen in training = 0
    n = t["n"].fillna(0).values
    mean = (t["y"].fillna(0).values + entity_smoothing * global_mean) / (n + entity_smoothing)
    mean_square = (t["y2"].fillna(0).values + entity_smoothing * global_mean_square) / (n + entity_smoothing)
    out = {f"{prefix}_mean": mean,
           f"{prefix}_std": np.sqrt(np.maximum(mean_square - mean ** 2, 0)),
           f"{prefix}_bias": t["r"].fillna(0).values / (n + entity_smoothing),
           f"{prefix}_count": n * count_scale}  # OOF see 80% of the data --> rescale
    # Rating distributions reveal whether albums or artists receive mixed scores
    if with_score_shares:
        for i in range(5):
            out[f"{prefix}_share{i + 1}"] = ((t[f"c{i}"].fillna(0).values + entity_smoothing * global_shares[i])
                                             / (n + entity_smoothing))
    return pd.DataFrame(out)


for prefix, with_score_shares in [("reviewer", False), ("album", True), ("artist", True)]:
    train_ids = train[entity_columns[prefix]].values
    fold_features = []
    # Calculate training statistics using labels from other folds
    for fit_idx, val_idx in folds:
        part = entity_features(entity_totals(train_ids[fit_idx], fit_idx), train_ids[val_idx],
                               prefix, with_score_shares, count_scale=1.25)
        part.index = val_idx
        fold_features.append(part)
    x_train = pd.concat([x_train, pd.concat(fold_features).sort_index()], axis=1)
    all_totals = entity_totals(train_ids, np.arange(len(train)))
    x_test = pd.concat([x_test, entity_features(all_totals, test[entity_columns[prefix]].values,
                                                prefix, with_score_shares)], axis=1)

x_train["reviewer_has_history"] = (x_train.reviewer_count > 0).astype(int)
x_test["reviewer_has_history"] = (x_test.reviewer_count > 0).astype(int)

# Text only prediction corrected by the three biases
for x, base in [(x_train, text_score), (x_test, text_score_test)]:
    x["adjusted_text_score"] = base + x.reviewer_bias + x.album_bias + x.artist_bias

# Add earlier model predictions as features; combines text sentiment with reviewer and music metadata
for i in range(5):
    x_train[f"lr_prob_score{i + 1}"], x_test[f"lr_prob_score{i + 1}"] = oof_lr[:, i], test_lr[:, i]
if use_ridge:
    x_train["ridge_score"], x_test["ridge_score"] = oof_ridge, test_ridge
for j, name in enumerate(lexicon_columns):
    x_train[name], x_test[name] = oof_lexicon[:, j], test_lexicon[:, j]
for j, threshold in enumerate(thresholds):
    x_train[f"prob_score_ge{threshold}"] = oof_ordinal[:, j]
    x_test[f"prob_score_ge{threshold}"] = test_ordinal[:, j]
x_train["ordinal_total"], x_test["ordinal_total"] = oof_ordinal.sum(1), test_ordinal.sum(1)



# Train Random Forest and ExtraTrees on the combined features
print(f"Training forests on {x_train.shape[1]} features...")
forest_folds = StratifiedKFold(5, shuffle=True, random_state=1)

random_forest = RandomForestClassifier(
    n_estimators=rf_trees, min_samples_leaf=rf_min_leaf, max_features=rf_max_features,
    class_weight="balanced_subsample", n_jobs=-1, random_state=0)
oof_rf = cross_val_predict(random_forest, x_train, y, cv=forest_folds, method="predict_proba")
print("Random Forest OOF macro F1 (raw):", round(macro_f1(y, oof_rf.argmax(1) + 1), 4))

rf_weight = 1.0                                   # Share of the blend given to the Random Forest
if use_extra_trees:
    extra_trees = ExtraTreesClassifier(
        n_estimators=et_trees, min_samples_leaf=et_min_leaf, max_features=et_max_features,
        class_weight="balanced", n_jobs=-1, random_state=0)
    oof_et = cross_val_predict(extra_trees, x_train, y, cv=forest_folds, method="predict_proba")
    print("ExtraTrees OOF macro F1 (raw):", round(macro_f1(y, oof_et.argmax(1) + 1), 4))

    best_blend_score = -1
    for weight in np.arange(0, 1.01, 0.1):
        blend_score = macro_f1(y, (weight * oof_rf + (1 - weight) * oof_et).argmax(1) + 1)
        print(f"  blend weight on Random Forest = {weight:.1f}: {blend_score:.4f}")
        if blend_score > best_blend_score:
            best_blend_score, rf_weight = blend_score, weight
    print(f"Using weight {rf_weight:.1f} on the Random Forest")
    oof_blend = rf_weight * oof_rf + (1 - rf_weight) * oof_et
else:
    oof_blend = oof_rf



# Tune class probabilities to improve macro F1
print("Tuning class multipliers...")
multipliers = np.ones(5)
best_score = macro_f1(y, oof_blend.argmax(1) + 1)
print("Macro F1 before multipliers:", round(best_score, 4))
for _ in range(3):                                #Coordinate search
    for c in range(5):
        for candidate in np.arange(0.1, 3.0, 0.05):
            trial = multipliers.copy()
            trial[c] = candidate
            trial_score = macro_f1(y, (oof_blend * trial).argmax(1) + 1)
            if trial_score > best_score:
                best_score, multipliers = trial_score, trial
print(f"\nFinal Macro F1: {best_score:.4f}")
print("Class multipliers:", np.round(multipliers, 2))
print("\nPer class report (out of fold):")
print(classification_report(y, (oof_blend * multipliers).argmax(1) + 1, digits=3))

# Final fit and predictions
random_forest.fit(x_train, y)
test_probs = rf_weight * random_forest.predict_proba(x_test)
if use_extra_trees and rf_weight < 1:             # skip the fit if ExtraTrees has zero weight
    test_probs += (1 - rf_weight) * extra_trees.fit(x_train, y).predict_proba(x_test)

importances = pd.Series(random_forest.feature_importances_, index=x_train.columns)
print("Top 20 features (Random Forest):\n", importances.sort_values(ascending=False).head(20).round(4).to_string())

predictions = (test_probs * multipliers).argmax(1) + 1
pd.DataFrame({"id": test.id, "Score": predictions}).to_csv("submission.csv", index=False)
print(f"\nTotal time: {time.time() - start_time:.1f}s")
print("Saved predictions to 'submission.csv'")