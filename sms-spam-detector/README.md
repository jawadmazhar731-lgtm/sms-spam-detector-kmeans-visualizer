# SMS Spam Detector — Naive Bayes ML Project

Machine Learning mids project (Iqra University, ML Lab — Lab 4: Naive Bayes Classifier).
Classifies SMS messages as spam or ham.

## Contents

- `sms_spam_detector.py` — main classifier (baseline + improved TF-IDF version)
- `SMSSpamCollection` — dataset file (UCI SMS Spam Collection, 5,572 messages)
- `app.py` — simple web UI demo for live spam/ham checks
- `IMPROVEMENTS.md` — what was improved over the baseline and why, with results
- `windows-app/` — packaged Windows version (trainer + runner scripts, requirements)
- `app-screenshots/` — real output screenshots

## Results

- Baseline accuracy: 98.74%
- Improved model accuracy: 98.39% (with TF-IDF, stopwords, stemming, stratified split)
- 5-fold cross-validation: 95.15%

## Run

```bash
pip install -r windows-app/requirements.txt
python3 sms_spam_detector.py
```

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Tech

Python, scikit-learn (Multinomial Naive Bayes, TF-IDF), Flask for the demo app.
