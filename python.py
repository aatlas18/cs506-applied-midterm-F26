import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/train.csv")
sample = pd.read_csv("data/sample_submission.csv")

train = df[df["Score"].notna()].copy().reset_index(drop=True)
test = df[df["Score"].isna()].copy().reset_index(drop=True)

genres = [
    "rock", "pop", "jazz", "metal", "electronic",
    "hip hop", "folk", "country", "classical", "blues"
]

keys = ["reviewerID", "artist_mbid", "album_mbid"]

# Create features from the reviews
def add_features(data):
    data = data.copy()

    summary = data["summary"].fillna("").astype(str)
    review = data["reviewText"].fillna("").astype(str)

    data["summary_text"] = summary
    data["review_text"] = review

    data["year"] = pd.to_datetime(
        data["unixReviewTime"], unit="s", errors="coerce"
    ).dt.year

    data["length"] = np.log1p(review.str.len())
    data["exclamations"] = review.str.count("!")
    data["capital_words"] = review.str.count(r"[A-Z]{3,}")

    data["summary_length"] = np.log1p(summary.str.len())
    data["summary_words"] = np.log1p(summary.str.split().str.len())
    data["summary_exclamations"] = summary.str.count("!")
    data["summary_questions"] = summary.str.count(r"\?")

    votes = pd.to_numeric(
        data["TotalVotes"], errors="coerce"
    ).fillna(0).clip(lower=0)

    helpful = pd.to_numeric(
        data["VotedHelpful"], errors="coerce"
    ).fillna(0).clip(lower=0)

    data["helpful_ratio"] = helpful / (votes + 1)
    data["log_votes"] = np.log1p(votes)

    genre_text = data["genres"].fillna("").astype(str).str.lower()

    data["genre_count"] = genre_text.apply(
        lambda x: len([g for g in x.split(",") if g.strip()])
    )

    for genre in genres:
        name = "genre_" + genre.replace(" ", "_")
        data[name] = genre_text.str.contains(genre, regex=False).astype(int)

    return data


train = add_features(train)
test = add_features(test)

all_data = pd.concat([train, test], ignore_index=True)

for col in ["reviewerID", "artist_mbid"]:
    counts = all_data[col].value_counts()
    train[col + "_count"] = train[col].map(counts).fillna(0)
    test[col + "_count"] = test[col].map(counts).fillna(0)


# Average ratings for each reviewer artist and album
def target_encoding(source, target, col):
    mean = source["Score"].mean()
    stats = source.groupby(col)["Score"].agg(["sum", "count"])

    averages = (stats["sum"] + 10 * mean) / (stats["count"] + 10)

    values = target[col].map(averages).fillna(mean).to_numpy()
    counts = target[col].map(stats["count"]).fillna(0).to_numpy()

    return values, counts


# Combine metadata and text model predictions
def make_features(source, target, probabilities):
    cols = [
        "year", "length", "exclamations", "capital_words",
        "helpful_ratio", "log_votes", "genre_count",
        "reviewerID_count", "artist_mbid_count",
        "summary_length", "summary_words",
        "summary_exclamations", "summary_questions"
    ]

    cols += ["genre_" + genre.replace(" ", "_") for genre in genres]

    X = target[cols].reset_index(drop=True).copy()

    for i in range(5):
        X[f"p{i + 1}"] = probabilities[:, i]

    for col in keys:
        values, counts = target_encoding(source, target, col)
        X[col + "_te"] = values
        X[col + "_te_count"] = counts

    return X

# Split the labeled reviews
train_idx, val_idx = train_test_split(
    np.arange(len(train)),
    test_size=0.2,
    stratify=train["Score"],
    random_state=0
)

train_data = train.iloc[train_idx].reset_index(drop=True)
val_data = train.iloc[val_idx].reset_index(drop=True)

y_train = train_data["Score"].astype(int).to_numpy()
y_val = val_data["Score"].astype(int).to_numpy()


# TF-IDF features for summaries and full reviews
print("Creating TF-IDF features", flush=True)

summary_vectorizer = TfidfVectorizer(
    min_df=3,
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype=np.float32
)

review_vectorizer = TfidfVectorizer(
    min_df=3,
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype=np.float32
)

X_summary = summary_vectorizer.fit_transform(train_data["summary_text"])
X_review = review_vectorizer.fit_transform(train_data["review_text"])

X_val_summary = summary_vectorizer.transform(val_data["summary_text"])
X_val_review = review_vectorizer.transform(val_data["review_text"])

# Give summaries more weight
X_text = hstack([X_summary * 2, X_review], format="csr")
X_val_text = hstack([X_val_summary * 2, X_val_review], format="csr")

# Train the text model using five folds
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(train_data), 5))

print("Training text model", flush=True)

for fold, (a, b) in enumerate(folds.split(X_text, y_train), 1):
    print(f"Fold {fold}/5", flush=True)

    model = LogisticRegression(C=2, max_iter=300)
    model.fit(X_text[a], y_train[a])
    oof_probs[b] = model.predict_proba(X_text[b])

text_model = LogisticRegression(C=2, max_iter=300)
text_model.fit(X_text, y_train)

text_val_probs = text_model.predict_proba(X_val_text)

print(
    "Text Macro F1:",
    round(f1_score(y_val, text_model.predict(X_val_text), average="macro"), 4)
)

# Add target encodings without the row's own rating
X_train = make_features(train_data, train_data, oof_probs)

for col in keys:
    values = np.zeros(len(train_data))
    counts = np.zeros(len(train_data))

    for a, b in folds.split(X_text, y_train):
        values[b], counts[b] = target_encoding(
            train_data.iloc[a], train_data.iloc[b], col
        )

    X_train[col + "_te"] = values
    X_train[col + "_te_count"] = counts

X_val = make_features(train_data, val_data, text_val_probs)


# Handle missing values and scale features
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_val = X_val.replace([np.inf, -np.inf], np.nan)

medians = X_train.median().fillna(0)

X_train = X_train.fillna(medians).fillna(0)
X_val = X_val.fillna(medians).fillna(0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Train the second logistic regression
stack_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

stack_model.fit(X_train, y_train)
val_probs = stack_model.predict_proba(X_val)

print(
    "Stacked Macro F1:",
    round(f1_score(y_val, stack_model.predict(X_val), average="macro"), 4)
)

# Tune the class weights
rng = np.random.default_rng(42)
indices = rng.permutation(len(y_val))
tune_idx, check_idx = np.array_split(indices, 2)
weights = np.ones(5)
best_score = f1_score(
    y_val[tune_idx],
    stack_model.classes_[np.argmax(val_probs[tune_idx], axis=1)],
    average="macro"
)

for _ in range(3):
    for i in range(5):
        for weight in np.arange(0.5, 2.51, 0.1):
            trial = weights.copy()
            trial[i] = weight

            predictions = stack_model.classes_[
                np.argmax(val_probs[tune_idx] * trial, axis=1)
            ]

            score = f1_score(y_val[tune_idx], predictions, average="macro")

            if score > best_score:
                best_score = score
                weights = trial

# Evaluate final weights
before = stack_model.classes_[np.argmax(val_probs[check_idx], axis=1)]
after = stack_model.classes_[
    np.argmax(val_probs[check_idx] * weights, axis=1)
]

before_f1 = f1_score(y_val[check_idx], before, average="macro")
after_f1 = f1_score(y_val[check_idx], after, average="macro")

print("\nResults")
print("Class weights:", np.round(weights, 2))
print("Macro F1 before tuning:", round(before_f1, 4))
print("Macro F1 after tuning:", round(after_f1, 4))

print("\nClassification Report:")
print(classification_report(y_val[check_idx], after, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val[check_idx], after))


# Predict scores for the unlabeled reviews
X_test_summary = summary_vectorizer.transform(test["summary_text"])
X_test_review = review_vectorizer.transform(test["review_text"])

X_test_text = hstack(
    [X_test_summary * 2, X_test_review],
    format="csr"
)

test_probs = text_model.predict_proba(X_test_text)

X_test = make_features(train_data, test, test_probs)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(medians).fillna(0)
X_test = scaler.transform(X_test)

final_probs = stack_model.predict_proba(X_test)

predictions = stack_model.classes_[
    np.argmax(final_probs * weights, axis=1)
].astype(int)


# Save the submission file
lookup = pd.Series(predictions, index=test["id"])

submission = sample.copy()
submission["Score"] = submission["id"].map(lookup)

assert len(submission) == 80000
assert submission["Score"].notna().all()
assert submission["Score"].isin([1, 2, 3, 4, 5]).all()

submission["Score"] = submission["Score"].astype(int)
submission.to_csv("submission.csv", index=False)

print("\nSubmission saved")
print(submission["Score"].value_counts().sort_index())