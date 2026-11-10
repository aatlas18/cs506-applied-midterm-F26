# CS506 Applied Midterm F26

This is the code and write up for the CS506 Applied Midterm, which strived to predict reviewer ratings (1-5) of music releases (albums, EPs, etc.), scored on macro F1.

**Kaggle Username**:  aatlas18

**Final Prediction Accuracy**: 
0.57036 (30th Place)

--------
## 1. Data Exploration
I started by exploring the training data using data_exploration.py to better understand the features and identify relationships that could improve my predictions. All visualizations are saved in eda_plots/.

Further explanation of each of the graphs:
1. Missing values
    - About 18% of the Score column was missing, representing the reviews we needed to predict. The other columns were essentially complete.
2. Score distribution
    - Shows how many reviews received each score
    - The data was heavily imbalanced, with almost half of reviews receiving 5 stars (48.7%) and only 6.4% receiving 2 stars. 
    - Since macro F1 weights every class equally, I needed to account for the less common ratings. This is why the model puts so much effort into the rare classes.
3. Mean score across numeric features
    - Shows each numeric feature split into deciles, with the mean score of each bin. The dashed line is the overall mean score (about 3.92).
    -These features are used in basic_features.
4. Score Boxplots
    - Shows the distribution of review length, summary length, and exclamation count for each score.
    - Scores 2 and 3 had the longest reviews, while 5-star reviews were shortest. Since length alone was not a strong predictor, I used TF-IDF text features alongside length.
5. Reviews and mean score over time 
    - Shows: The number of reviews per year (gray bars) and the mean score per year (red line).
    - I include the review timestamp (unixReviewTime) as a feature because tree models can capture the nonlinear changes in ratings over time.
6. Genres
    - Shows: the most common genres (left) and the mean score for each genre (right). The dashed line is the overall mean.
    - In my model, I add an indicator for each of the 25 most common genres, plus the number of genres per release, so the model can account for each genre's typical score.
7. Reviewer behaviour
    - Shows: How many reviews each reviewer wrote (left), the mean score of reviewers with at least 5 reviews (middle), and how much each of those reviewers' scores vary (right).
    - Each reviewer's history is a strong signal, so I compute reviewer, album, and artist statistics: smoothed mean score, standard deviation, review count, and bias against the text prediction.
8. Good and bad words
    - Shows: The 20 terms (single words and two-word phrases appearing at least 100 times) whose reviews have the lowest and highest mean scores.
    - Many of the strongest terms are two-word phrases, and their meaning depends on the pairing ("very disappointed" is much more negative than "disappointed"). I used this in the TF-IDF features, so I included bigrams and calculated word-polarity features.
9. Term polarity vs. frequency
    - Shows: Each term's mean score plotted against how often it appears (on a log scale).
    - The learned lexicon smooths every word's polarity toward the overall mean (lexicon_smoothing = 20), so a word seen only a few times can't claim an extreme polarity.
10. Correlation matrix
    - Shows: Spearman (rank) correlations between every pair of engineered features.
    - I used this to reduce redundant features, keeping text length instead of word count and using unhelpful votes and net sentiment counts
11. Correlation with score
    - Shows: Each feature's Spearman correlation with the score, ranked.
    - Because no single hand-made feature predicts the score well, the model relies on the TF-IDF text models for most of its signal and uses these features as supporting inputs. 


## 2. Feature Extraction / Engineering
I built the features in two groups: text-based predictions, and handmade/metadata features. 

- Text model features
    - TF-IDF: Extracted unigrams and bigrams from review text (300,000 features) and summaries (100,000 features).
    - Logistic Regression probabilities: Predicted the probability of each of the five ratings.
    - Ridge score: Predicted a continuous rating to account for the order of scores.
    - Ordinal probabilities: Predicted whether a review received at least 2, 3, 4, or 5 stars.
    - Word polarity: Calculated sentiment scores for words and phrases based on their average associated ratings.
- Feature Engineering
    - Total votes, unhelpful votes, helpfulness ratio, and review timestamp.
    - Writing style: Review and summary length, exclamation marks, question marks, and capital letter ratio.
    - Sentiment words: Counts of hand picked good words, bad words, and negations.
    - Written ratings: Many reviewers state their rating in the text ("4/5", "8 out of 10", "two stars"). I extract these, convert them to a 0-5 scale, and add a flag for whether one was found.
    - Genres: Indicators for the 25 most common genres.
    - Reviewer, album, and artist review counts and rating statistics.
    - Smoothed average ratings and differences between actual ratings and text predictions for reviewers, albums, and artists.

## 3. Model Creation & Assumptions
- The final model in two stages:
    - Stage 1: I trained Logistic Regression, Ridge Regression, and ordinal Logistic Regression models using TF-IDF features. I also created word polarity features.
    - Stage 2: I combined these predictions with the additional features and trained a Random Forest and Extra Trees classifier. I tested blending their probabilities and adjusted the final predictions using class-specific multipliers.
- Assumptions:
    - My main assumptions were that review text be the strongest, ratings have a natural order, and rating patterns would be consistent. I also expected tree-based models to capture relationships that linear models might miss.
    - To prevent data leakage, I used 5-fold stratified cross-validation. All training features that depended on ratings were calculated out of fold, meaning each fold's features were generated without using its own labels. For the test data, I used the full labeled training set.

## 4. Model Tuning
- Random Forest: 400 trees, minimum leaf size 5, sqrt features per split, class balanced subsamples.
- Extra Trees: 400 trees, minimum leaf size 3, 30% of features per split, class balanced.
- I also tested different blending weights. Random Forest performed slightly better on its own (0.5547 macro F1) than Extra Trees (0.5513). Since blending did not improve the score, I ultimately used Random Forest alone. 
    - Extra Trees only (weight 0.0): 0.5513
    - Even split (weight 0.5): 0.5537
    - Peak (weight 0.7): 0.5547
    - Weights 0.8 and 0.9: 0.5546 and 0.5545, a slight dip
    - Random Forest only (weight 1.0): 0.5547, tied with the peak and chosen
- I then adjusted the class probabilities to improve predictions for the less common ratings. I tuned a multiplier on each class's probability with coordinate search, shifting predictions toward classes the model underpredicts.
    - Score 1: 0.65 (lowered)
    - Score 2: 1.05 (raised)
    - Score 3: 1.05 (raised)
    - Score 4: 0.95 (lowered)
    - Score 5: 0.85 (lowered)
- These adjustments helped the model predict more middle ratings instead of favoring the more common or extreme scores. This increased my out-of-fold macro F1 from 0.5547 to 0.5623.

## 5. Model Evaluation /  Performance
I evaluated my model using 5-fold stratified cross-validation and macro F1. The final model was trained on 367,583 labeled reviews and predicted ratings for 80,000 test reviews.

- **Out-of-fold macro F1:** 0.5623
- **Out-of-fold accuracy:** 0.659
- **Kaggle macro F1:** **0.57036 (30th place)**

My Kaggle score was slightly higher than my out-of-fold score, suggesting that the model performed well on unseen data. However, the out-of-fold score may be a little optimistic since I used the same predictions for both tuning and evaluation.

---
### Per-class results (out of fold)
- Score 1 (39,353 reviews): Precision 0.826, recall 0.682, F1 0.747
- Score 2 (23,476 reviews): Precision 0.317, recall 0.386, F1 0.348
- Score 3 (44,933 reviews): Precision 0.363, recall 0.404, F1 0.382
- Score 4 (80,708 reviews): Precision 0.481, recall 0.508, F1 0.494
- Score 5 (179,113 reviews): Precision 0.859, recall 0.821, F1 0.840
- Macro average: Precision 0.569, recall 0.560, F1 0.562
- Weighted average: Precision 0.677, recall 0.659, F1 0.667
- Accuracy: 0.659 across all 367,583 reviews

The model performed best on 1-star and 5-star reviews, which makes sense because these reviews often use more clearly positive or negative language. The biggest challenge was predicting scores of 2 and 3. These reviews often contain a mix of positive and negative opinions, making them harder to distinguish. Since macro F1 treats every class equally, the weaker performance on these middle ratings lowered my overall score.

## 6. Struggles / Issues/ Open Questions
- Data Leakage: Since I used reviewer, album, and artist rating statistics, I needed to calculate these features out of fold to avoid giving the model information it would not have for new reviews.
- Runtime and memory: The TF-IDF matrices were large, and retraining the text models took considerable time. To make experimentation more efficient, I cached intermediate results in cache/ and limited parallel processing.
- Middle scores: I also struggled with predicting middle ratings. Even after adjusting the class probabilities, scores of 2 and 3 remained the hardest to classify.
- Tuning on the same folds: The blend weight and class multipliers are tuned on the same out of fold predictions they are scored on, so the out of fold score may be slightly optimistic.
- Extra Trees added nothing: I expected blending two forests to help, but the blend search gave Extra Trees zero weight. A more different second model would likely blend better.
- Overall, I think the model performed reasonably well, but there is still room for improvement, particularly in distinguishing between middle ratings. One question I would explore further is whether different text features or another model covered in class could improve these predictions. 
- Open Questions: Could different text features or another model covered in class improve predictions, especially for middle ratings?

## References
1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
2. scikit-learn developers. `TfidfVectorizer`. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
3. scikit-learn developers. `LogisticRegression`. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
4. scikit-learn developers. `Ridge`. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
5. scikit-learn developers. `RandomForestClassifier`. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
6. scikit-learn developers. `ExtraTreesClassifier`. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
7. scikit-learn developers. `StratifiedKFold`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
8. scikit-learn developers. `cross_val_predict`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html