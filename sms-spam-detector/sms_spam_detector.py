# ML Lab Mids Project - SMS Spam Detector
# Topic: Lab 4 - Naive Bayes Classifier
# Student: Jawad Mazhar, Iqra University (ML Lab)
# Original source (dataset): UCI SMS Spam Collection
# https://archive.ics.uci.edu/dataset/228/sms+spam+collection
#
# What this program does:
# It reads SMS messages, learns which messages are Spam and which are Ham
# (normal), and then predicts Spam or Ham for any new message you type.
#
# How to run:
#   python3 sms_spam_detector.py
# You need: pandas, scikit-learn. The file SMSSpamCollection must be
# in the same folder as this program.

import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# A small list of common English stopwords. These words appear in almost
# every message, so they do not help us tell Spam from Ham.
STOPWORDS = set([
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "our", "their", "and", "or", "but", "if", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "this", "that", "it",
])


def simple_stem(word):
    """A very simple stemmer: cut common endings so related words match.

    Example: winning -> winn, playing -> play. We keep it gentle (only
    'ing' and 'ed') so words are not cut too much. This keeps the code
    easy to explain in viva without needing any extra library.
    """
    for ending in ("ing", "ed"):
        if len(word) > len(ending) + 2 and word.endswith(ending):
            return word[:-len(ending)]
    return word


def clean_text(text):
    """Step 1 - Data cleaning, done in this order:
    1. Lowercase: 'FREE' and 'free' become the same word.
    2. Remove numbers and punctuation, keep only letters and spaces.
    3. Remove stopwords.
    4. Stemming: 'winning' -> 'win'.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = []
    for word in text.split():
        if word not in STOPWORDS:
            words.append(simple_stem(word))
    return " ".join(words)


def show_results(name, y_true, y_pred):
    """Print accuracy, precision, recall, F1 and the confusion matrix."""
    print("\n---", name, "---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average="weighted"), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average="weighted"), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred, average="weighted"), 4))
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print("               predicted ham   predicted spam")
    matrix = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    print("actual ham    ", matrix[0])
    print("actual spam   ", matrix[1])
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def main():
    # Step 2 - Load the real dataset. It has two columns, separated by
    # a tab: label (ham/spam) and the message text. No header row.
    data = pd.read_csv("SMSSpamCollection", sep="\t",
                       header=None, names=["label", "message"])
    print("Total messages in dataset:", len(data))
    print("Ham messages:", (data["label"] == "ham").sum())
    print("Spam messages:", (data["label"] == "spam").sum())

    # Step 3 - Split the data: 80% for training, 20% for testing.
    # stratify keeps the same ham/spam ratio in both parts.
    # random_state=42 makes the split the same every time we run it.
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"],
        test_size=0.20, random_state=42, stratify=data["label"])

    # BASELINE MODEL (the original, simple version):
    # Simple word counts + Naive Bayes, no cleaning at all.
    baseline_vectorizer = CountVectorizer()
    X_train_base = baseline_vectorizer.fit_transform(X_train)
    X_test_base = baseline_vectorizer.transform(X_test)
    baseline_model = MultinomialNB()
    baseline_model.fit(X_train_base, y_train)
    baseline_pred = baseline_model.predict(X_test_base)
    show_results("BASELINE: CountVectorizer + Naive Bayes", y_test, baseline_pred)

    # IMPROVED MODEL (our improvement):
    # Clean the messages first, then use TF-IDF instead of simple counts.
    # TF-IDF gives a higher weight to rare, important words like 'free'
    # or 'prize', and a lower weight to common words.
    cleaned_train = X_train.apply(clean_text)
    cleaned_test = X_test.apply(clean_text)
    # ngram_range=(1, 2) means single words AND word pairs are used,
    # so phrases like 'free prize' are also learned, not just 'free'.
    # alpha=0.1 in the first run was tried, alpha=0.05 gave the best
    # result here, so we use alpha=0.05 (less smoothing).
    improved_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_new = improved_vectorizer.fit_transform(cleaned_train)
    X_test_new = improved_vectorizer.transform(cleaned_test)
    improved_model = MultinomialNB(alpha=0.05)
    improved_model.fit(X_train_new, y_train)
    improved_pred = improved_model.predict(X_test_new)
    show_results("IMPROVED: TF-IDF + Cleaning + Naive Bayes", y_test, improved_pred)

    # Cross-validation: test the improved model 5 times on different
    # parts of the training data, so the result is reliable, not luck.
    scores = cross_val_score(MultinomialNB(alpha=0.05), X_train_new, y_train, cv=5)
    print("\nCross-Validation (5-fold) accuracy scores:", [round(s, 4) for s in scores])
    print("Mean cross-validation accuracy:", round(scores.mean(), 4))

    # LIVE PREDICTION: type any message, get Spam or Ham.
    def predict_message(message):
        cleaned = clean_text(message)
        vector = improved_vectorizer.transform([cleaned])
        return improved_model.predict(vector)[0]

    print("\n--- LIVE DEMO: type a message, get Spam or Ham ---")
    demo_messages = [
        "Congratulations! You have won a FREE prize, click this link now to claim",
        "WINNER! You won 1000 cash, call now to collect your prize",
        "Hi, are we meeting for class tomorrow at 8:30?",
        "Ok, I will bring the lab manual tomorrow",
    ]
    for message in demo_messages:
        print(f"\nMessage: {message}")
        print("Prediction:", predict_message(message))

    # Interactive part: keep asking until the user types 'exit'.
    print("\nNow you try! Type a message (or 'exit' to stop):")
    while True:
        try:
            user_message = input("> ")
        except EOFError:
            break
        if user_message.strip().lower() == "exit":
            break
        if user_message.strip():
            print("Prediction:", predict_message(user_message))


if __name__ == "__main__":
    main()
