import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning)
t0 = time.time()

# Config
USE_RIDGE = True          # ordinal stage-1 model (predicts the score as a number)
SHOW_EXPLORATION = False  # word printout (data_exploration.py already covers this)
N_JOBS = 3                # parallel CV folds for LR/Ridge; set to 1 if you run low on RAM
LEX_K = 20                # word-polarity smoothing (higher = shrink rare words more)
LEX_THR = 0.30            # |polarity| (score points) for a term to count as good / bad
TOP_N_GENRES = 25         # multi-hot genre columns
RF_TREES, RF_LEAF, RF_MAX_FEATURES = 400, 5, "sqrt"
TAG = f"v3_r{int(USE_RIDGE)}_k{LEX_K}_t{LEX_THR}"
CACHE = f"cache/stage1_{TAG}.npz"
os.makedirs("cache", exist_ok=True)

# Load
print("Loading datasets...")
df = pd.read_csv("data/train.csv")
lab = df[df.Score.notna()].reset_index(drop=True)
unl = df[df.Score.isna()].reset_index(drop=True)
y = lab.Score.astype(int).values
print(f"  labeled={len(lab)}  to-predict={len(unl)}")

if os.path.exists("data/test.csv"):
    tid = set(pd.read_csv("data/test.csv").iloc[:, 0])
    print("  test.csv ids match the unlabeled rows:", tid == set(unl.id))

skf = StratifiedKFold(5, shuffle=True, random_state=0)
folds = list(skf.split(lab, y))


def fast_macro_f1(y_true, y_pred, k=5):
    """Macro F1 from a confusion matrix (much faster than sklearn inside tuning loops)."""
    cm = np.bincount((y_true - 1) * k + (y_pred - 1), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return (2 * prec * rec / np.maximum(prec + rec, 1e-12)).mean()


# Wrd-polarity lexicon
def word_polarity(B, yy, k=LEX_K):
    """Smoothed (mean score of reviews containing the term) - (global mean)."""
    g = yy.mean()
    cnt = np.asarray(B.sum(0)).ravel()
    sm = np.asarray(B.T.dot(yy.astype(float))).ravel()
    return (sm + k * g) / (cnt + k) - g, cnt


LEX_COLS = ["lex_mean", "lex_max", "lex_min", "lex_good", "lex_bad",
            "lex_good_frac", "lex_bad_frac", "lex_net", "lex_strong_good", "lex_strong_bad"]


def lex_features(B, pol, thr=LEX_THR):
    n = np.maximum(B.getnnz(axis=1), 1).astype(float)
    M = B.multiply(pol[None, :]).tocsr()
    mean = np.asarray(M.sum(1)).ravel() / n
    mx = M.max(axis=1).toarray().ravel()
    mn = M.min(axis=1).toarray().ravel()
    good = np.asarray(B.dot((pol > thr).astype(float))).ravel()
    bad = np.asarray(B.dot((pol < -thr).astype(float))).ravel()
    sg = np.asarray(B.dot((pol > 2 * thr).astype(float))).ravel()
    sb = np.asarray(B.dot((pol < -2 * thr).astype(float))).ravel()
    return np.column_stack([mean, mx, mn, good, bad, good / n, bad / n, good - bad, sg, sb])


# Text models (TF-IDF + Logistic Regression / Ridge)
if os.path.exists(CACHE):
    print(f"Loading cached stage-1 outputs from {CACHE}  (delete cache/ to rebuild)")
    z = np.load(CACHE)
    oof_lr, te_lr, oof_rd, te_rd, lex_l, lex_u = (z[k] for k in
                                                  ["oof_lr", "te_lr", "oof_rd", "te_rd", "lex_l", "lex_u"])
else:
    print("Vectorizing text (review and summary get separate TF-IDF blocks)...")
    tf_rev = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000,
                             sublinear_tf=True, strip_accents="unicode")
    tf_sum = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                             sublinear_tf=True, strip_accents="unicode")
    Xl = hstack([tf_rev.fit_transform(lab.reviewText.fillna("")),
                 tf_sum.fit_transform(lab.summary.fillna(""))], format="csr")
    Xu = hstack([tf_rev.transform(unl.reviewText.fillna("")),
                 tf_sum.transform(unl.summary.fillna(""))], format="csr")
    vocab = np.concatenate([tf_rev.get_feature_names_out(),
                            np.char.add("[sum] ", tf_sum.get_feature_names_out().astype(str))])

    Bl = Xl.copy(); Bl.data[:] = 1.0      # binary "term present" matrices for the lexicon
    Bu = Xu.copy(); Bu.data[:] = 1.0

    if SHOW_EXPLORATION:
        pol, cnt = word_polarity(Bl, y)
        ex = pd.DataFrame({"term": vocab, "mean_score": pol + y.mean(), "n": cnt.astype(int)})
        ex = ex[ex.n >= 50].sort_values("mean_score")
        print("\nWorst terms:\n", ex.head(20).to_string(index=False))
        print("\nBest terms:\n", ex.tail(20).iloc[::-1].to_string(index=False))

    print("Building out-of-fold lexicon features...")
    lex_l = np.zeros((len(lab), len(LEX_COLS)))
    for tr, va in folds:
        pol, _ = word_polarity(Bl[tr], y[tr])
        lex_l[va] = lex_features(Bl[va], pol)
    pol_all, _ = word_polarity(Bl, y)
    lex_u = lex_features(Bu, pol_all)

    print("Training Logistic Regression (5-fold OOF)...")
    lr = LogisticRegression(C=1.0, max_iter=400, tol=1e-3, class_weight="balanced")
    oof_lr = cross_val_predict(lr, Xl, y, cv=skf, method="predict_proba", n_jobs=N_JOBS)
    lr.fit(Xl, y)
    te_lr = lr.predict_proba(Xu)

    if USE_RIDGE:
        print("Training Ridge (5-fold OOF)...")
        rd = Ridge(alpha=2.0)
        oof_rd = cross_val_predict(rd, Xl, y, cv=skf, n_jobs=N_JOBS)
        rd.fit(Xl, y)
        te_rd = rd.predict(Xu)
    else:
        oof_rd, te_rd = np.zeros(len(lab)), np.zeros(len(unl))

    np.savez(CACHE, oof_lr=oof_lr, te_lr=te_lr, oof_rd=oof_rd, te_rd=te_rd, lex_l=lex_l, lex_u=lex_u)
    print(f"  cached stage-1 outputs -> {CACHE}")
    del Xl, Xu, Bl, Bu

# Features
print("Building metadata features...")
allx = pd.concat([lab, unl], ignore_index=True)    # label-free counts only -> no leakage
cnts = {c: allx[c].map(allx[c].value_counts()) for c in ["reviewerID", "album_mbid", "artist_mbid"]}
n_lab = len(lab)

TOP_GENRES = (allx.genres.dropna().str.split(",").explode().str.strip()
              .value_counts().head(TOP_N_GENRES).index.tolist())
print(f"Top {TOP_N_GENRES} genres:", TOP_GENRES)

GOOD_SEED = ["masterpiece", "brilliant", "perfect", "classic", "amazing", "incredible", "essential",
             "stunning", "flawless", "gem", "favorite", "best", "excellent", "great", "love",
             "beautiful", "genius", "superb"]
BAD_SEED = ["boring", "disappointing", "disappointed", "awful", "terrible", "worst", "bland",
            "mediocre", "forgettable", "waste", "weak", "dull", "meh", "generic", "filler", "mess",
            "unlistenable", "overrated", "bad", "poor"]
good_re = re.compile(r"\b(?:" + "|".join(GOOD_SEED) + r")\b")
bad_re = re.compile(r"\b(?:" + "|".join(BAD_SEED) + r")\b")
neg_re = re.compile(r"\b(?:not|no|never|without|hardly)\b|n't")

# Explicit rating mentions: "4/5", "8 out of 10", "two stars", "5 star"
STAR_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
FRAC_RE = r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*(?:/|out of)\s*(5|10)\b"
STAR_RE = r"\b(zero|one|two|three|four|five|[0-5](?:\.5)?)[\s-]*stars?\b"


def star_features(low):
    m1 = low.str.extract(FRAC_RE)
    v1 = pd.to_numeric(m1[0], errors="coerce"); d1 = pd.to_numeric(m1[1], errors="coerce")
    r1 = (v1 / d1 * 5).where(v1 <= d1)
    m2 = low.str.extract(STAR_RE)[0]
    r2 = m2.map(STAR_WORDS).fillna(pd.to_numeric(m2, errors="coerce"))
    r = r1.fillna(r2).astype(float)
    return r.fillna(-1), r.notna().astype(int)


def meta(d, offset):
    f = pd.DataFrame(index=d.index)
    rt = d.reviewText.fillna(""); sm = d.summary.fillna("")
    low = (sm + " . " + rt).str.lower()
    f["tot"] = d.TotalVotes.fillna(0)
    f["unhelpful"] = (d.TotalVotes.fillna(0) - d.VotedHelpful.fillna(0)).clip(lower=0)
    f["ratio"] = d.VotedHelpful.fillna(0) / (f.tot + 1)
    f["len"] = rt.str.len()
    f["slen"] = sm.str.len()
    f["time"] = d.unixReviewTime.fillna(d.unixReviewTime.median())
    f["ngen"] = d.genres.fillna("").apply(lambda s: len([g for g in s.split(",") if g.strip()]))
    f["excl"] = rt.str.count("!")
    f["qmark"] = rt.str.count(r"\?")
    f["caps"] = rt.str.count(r"[A-Z]") / (f["len"] + 1)
    f["negations"] = low.str.count(neg_re)
    f["seed_bad"] = low.str.count(bad_re)
    f["seed_net"] = low.str.count(good_re) - f.seed_bad
    f["sum_bad"] = sm.str.lower().str.count(bad_re)
    f["star_val"], f["has_star"] = star_features(low)
    gsets = d.genres.fillna("").apply(lambda x: {g.strip() for g in x.split(",")})
    for g in TOP_GENRES:
        f["genre_" + re.sub(r"\W+", "_", g.lower()).strip("_")] = gsets.apply(lambda st, g=g: int(g in st))
    idx = np.arange(len(d)) + offset
    for c, nm in [("reviewerID", "reviewer_n"), ("album_mbid", "album_n"), ("artist_mbid", "artist_n")]:
        f[nm] = cnts[c].iloc[idx].values
    return f


Fl, Fu = meta(lab, 0), meta(unl, n_lab)

# OOF target statistics per key: mean, std, count, and score distribution
print("Calculating out-of-fold key statistics (reviewer / album / artist)...")
Y1 = np.eye(5)[y - 1]
G, G2, GP = y.mean(), (y.astype(float) ** 2).mean(), Y1.mean(0)


def key_stats(keys, rows):
    d = pd.DataFrame(Y1[rows], columns=[f"c{i}" for i in range(5)])
    d["y"] = y[rows]; d["y2"] = y[rows].astype(float) ** 2; d["n"] = 1.0; d["k"] = keys
    return d.groupby("k").sum()


def key_apply(st, keys, prefix, dist, cnt_scale=1.0, pw=5):
    s = st.reindex(keys)
    n = s["n"].fillna(0).values
    mean = (s["y"].fillna(0).values + pw * G) / (n + pw)
    m2 = (s["y2"].fillna(0).values + pw * G2) / (n + pw)
    out = {f"{prefix}_mean": mean,
           f"{prefix}_std": np.sqrt(np.maximum(m2 - mean ** 2, 0)),
           f"{prefix}_cnt": n * cnt_scale}          # OOF stats see 80% of data -> rescale counts
    if dist:
        for i in range(5):
            out[f"{prefix}_p{i + 1}"] = (s[f"c{i}"].fillna(0).values + pw * GP[i]) / (n + pw)
    return pd.DataFrame(out)


for col, prefix, dist in [("reviewerID", "rev", False), ("album_mbid", "alb", True),
                          ("artist_mbid", "art", True)]:
    k_all = lab[col].values
    parts = []
    for tr, va in folds:
        part = key_apply(key_stats(k_all[tr], tr), k_all[va], prefix, dist, cnt_scale=1.25)
        part.index = va
        parts.append(part)
    oof = pd.concat(parts).sort_index()
    te = key_apply(key_stats(k_all, np.arange(len(lab))), unl[col].values, prefix, dist)
    Fl = pd.concat([Fl, oof], axis=1)
    Fu = pd.concat([Fu, te], axis=1)
Fl["rev_hist"] = (Fl.rev_cnt > 0).astype(int)       # does this reviewer have any labeled history?
Fu["rev_hist"] = (Fu.rev_cnt > 0).astype(int)

for i in range(5):
    Fl[f"lr{i}"], Fu[f"lr{i}"] = oof_lr[:, i], te_lr[:, i]
if USE_RIDGE:
    Fl["ridge"], Fu["ridge"] = oof_rd, te_rd
for j, name in enumerate(LEX_COLS):
    Fl[name], Fu[name] = lex_l[:, j], lex_u[:, j]

# Random Forest
print(f"Training Random Forest on {Fl.shape[1]} features...")
rf = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=RF_LEAF,
                            max_features=RF_MAX_FEATURES, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=0)
oof_rf = cross_val_predict(rf, Fl, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                           method="predict_proba")
print("OOF macro F1 (raw):", round(fast_macro_f1(y, oof_rf.argmax(1) + 1), 4))

# Per class report (OOF, raw)
print("Tuning post-hoc class weight multipliers...")
w = np.ones(5)
best = fast_macro_f1(y, oof_rf.argmax(1) + 1)
for _ in range(3):
    for c in range(5):
        for cand in np.arange(0.1, 3.0, 0.05):
            w2 = w.copy(); w2[c] = cand
            s = fast_macro_f1(y, (oof_rf * w2).argmax(1) + 1)
            if s > best:
                best, w = s, w2
print(f"\nFinal Macro F1: {best:.4f}")
print("Optimal Class Multipliers:", np.round(w, 2))
print("\nPer-class report (OOF, tuned):")
print(classification_report(y, (oof_rf * w).argmax(1) + 1, digits=3))

rf.fit(Fl, y)
imp = pd.Series(rf.feature_importances_, index=Fl.columns).sort_values(ascending=False)
print("Top 20 features:\n", imp.head(20).round(4).to_string())

pred = (rf.predict_proba(Fu) * w).argmax(1) + 1
pd.DataFrame({"id": unl.id, "Score": pred}).to_csv("submission.csv", index=False)
print(f"\nTotal Time: {time.time() - t0:.1f}s")
print("Saved predictions to 'submission.csv'")