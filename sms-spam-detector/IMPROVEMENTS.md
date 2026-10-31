# Mids Project — Improvements Document
## ML Lab, Iqra University | Topic: Lab 4 — Naive Bayes Classifier

**Project:** SMS Spam Detector
**Student:** Jawad Mazhar
**Original source (dataset):** UCI SMS Spam Collection
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
**Code file:** `sms_spam_detector.py` (run it with `python3 sms_spam_detector.py`)
**Dataset file:** `SMSSpamCollection` (must be in the same folder as the code)

## 1. Original project (baseline)

A simple SMS spam classifier found in many online examples:

- CountVectorizer (simple word counts) + Multinomial Naive Bayes
- No text cleaning
- Only accuracy was reported in most simple versions

## 2. What we improved, and why

| # | Improvement | Why it helps |
|---|---|---|
| 1 | Data cleaning: lowercase, remove numbers/punctuation, remove stopwords, gentle stemming (ing/ed) | 'FREE' and 'free' become one word; useless words do not confuse the model |
| 2 | TF-IDF instead of simple word counts | Rare, important words like 'prize' get more weight than common words |
| 3 | Word pairs (ngram_range 1–2) | Phrases like 'free prize' are learned, not just single words |
| 4 | Tuned smoothing: MultinomialNB(alpha=0.05) | Less smoothing than the default, tested and chosen for this data |
| 5 | Stratified 80:20 split, random_state=42 | Same ham/spam ratio in train and test; same result every run |
| 6 | Full evaluation, not just accuracy | Precision, recall, F1-score and confusion matrix for both models |
| 7 | 5-fold cross-validation | Proves the result is reliable, not luck on one split |
| 8 | Live prediction demo | Type any message, get Spam/Ham instantly — used in viva demo |

## 3. Real results (from running the code, not estimated)

Dataset actually loaded: **5,572 messages — 4,825 ham, 747 spam**
Test set: 1,115 messages (966 ham, 149 spam)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline: CountVectorizer + Naive Bayes | 0.9874 | 0.9874 | 0.9874 | 0.9873 |
| Improved: TF-IDF + Cleaning + Naive Bayes | 0.9839 | 0.9839 | 0.9839 | 0.9835 |

Confusion matrix, improved model (rows = actual, columns = predicted):

|  | predicted ham | predicted spam |
|---|---|---|
| actual ham | 964 | 2 |
| actual spam | 16 | 133 |

Improved model, 5-fold cross-validation accuracy scores:
0.9484, 0.9664, 0.9506, 0.9473, 0.9450 — mean **0.9515**

## 4. Honest note (for viva)

On this dataset the improved model is **slightly below** the baseline
(98.39% vs 98.74% — a difference of 4 test messages out of 1,115).
TF-IDF does not always raise accuracy. The real improvements here are:
cleaning, phrase features, tuned smoothing, cross-validation,
full metrics, and a live demo the teacher can test. Saying this
honestly in viva is better than claiming a fake gain.

## 5. Live demo results (real predictions from the run)

- "Congratulations! You have won a FREE prize, click this link now to claim" → **spam**
- "WINNER! You won 1000 cash, call now to collect your prize" → **spam**
- "Hi, are we meeting for class tomorrow at 8:30?" → **ham**
- "Ok, I will bring the lab manual tomorrow" → **ham**

## 6. How to run

```
python3 sms_spam_detector.py
```

Then type any message when it asks, or type `exit` to stop.
You need pandas and scikit-learn installed.

## 7. Viva short answers

- **Naive Bayes kya hai?** Bayes theorem wala model, jo har lafz ki
  probability se faisla karta hai. 'Naive' is liye ke ye maan leta hai
  ke har lafz doosre se azad hai.
- **TF-IDF kya hai?** Important lafz ko zyada weight, aam lafz ko kam.
- **Improvement kya hai?** Cleaning, TF-IDF, word pairs, tuned alpha,
  cross-validation, full metrics aur live demo.
