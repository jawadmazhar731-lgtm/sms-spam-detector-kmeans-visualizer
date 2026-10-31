# SMS Spam Detector - UI App (ML Lab Mids Project, Lab 4 - Naive Bayes)
# Student: Jawad Mazhar, Iqra University (ML Lab)
# Original source (dataset): UCI SMS Spam Collection
# https://archive.ics.uci.edu/dataset/228/sms+spam+collection
#
# What this app does:
# A simple web UI for the spam detector built in sms_spam_detector.py.
# Type a message, click Predict, and the trained TF-IDF + Naive Bayes
# model (the real Python model, trained on the real UCI dataset) tells
# you Spam or Ham, with its confidence.
#
# How to run:
#   python3 app.py
# Then open http://localhost:8765 in a browser.
# Needs: pandas, scikit-learn (same as sms_spam_detector.py).
# No other package is needed - the web server is Python's built-in
# http.server, so the app runs on a normal student laptop.

import html
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Import the cleaning function from the original project code, so the
# app uses exactly the same cleaning as the trained model.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sms_spam_detector import clean_text  # noqa: E402

import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402

PORT = 8765
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "SMSSpamCollection")

# --- Train the real model once, when the app starts ---------------------
print("Loading dataset...", flush=True)
data = pd.read_csv(DATA_FILE, sep="\t", header=None,
                   names=["label", "message"])
TOTAL = len(data)
HAM = int((data["label"] == "ham").sum())
SPAM = int((data["label"] == "spam").sum())

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"],
    test_size=0.20, random_state=42, stratify=data["label"])

print("Training improved model (TF-IDF + Naive Bayes)...", flush=True)
cleaned_train = X_train.apply(clean_text)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_new = vectorizer.fit_transform(cleaned_train)
model = MultinomialNB(alpha=0.05)
model.fit(X_train_new, y_train)

# Real test results, computed from the same split as sms_spam_detector.py
cleaned_test = X_test.apply(clean_text)
y_pred = model.predict(vectorizer.transform(cleaned_test))
ACCURACY = float(accuracy_score(y_test, y_pred))
CM = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print(f"Dataset: {TOTAL} messages ({HAM} ham, {SPAM} spam)", flush=True)
print(f"Improved model test accuracy: {ACCURACY:.4f}", flush=True)
print(f"Confusion matrix (ham/spam): {CM.tolist()}", flush=True)


def predict(message):
    """Predict with the trained model. Returns (label, spam_probability)."""
    vector = vectorizer.transform([clean_text(message)])
    label = model.predict(vector)[0]
    probs = model.predict_proba(vector)[0]
    spam_index = list(model.classes_).index("spam")
    return label, float(probs[spam_index])


SAMPLE_SPAM = "Congratulations! You have won a FREE prize, click this link now to claim"
SAMPLE_HAM = "Hi, are we meeting for class tomorrow at 8:30?"

CSS = """
body { font-family: Arial, sans-serif; background: #f2f4f7; margin: 0; }
.wrap { max-width: 720px; margin: 40px auto; background: #fff; border-radius: 14px;
        padding: 28px 30px; box-shadow: 0 4px 18px rgba(0,0,0,.12); }
h1 { margin: 0 0 4px; color: #111; font-size: 28px; }
.sub { color: #555; margin: 0 0 20px; }
textarea { width: 100%; height: 90px; font-size: 16px; padding: 10px;
           border: 2px solid #ccc; border-radius: 8px; box-sizing: border-box; }
.row { margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }
button, .sample { background: #0a7a42; color: #fff; border: 0; border-radius: 8px;
    padding: 12px 22px; font-size: 16px; cursor: pointer; text-decoration: none; }
.sample { background: #e8eef5; color: #123; }
.result { margin-top: 20px; padding: 18px; border-radius: 10px; font-size: 20px; }
.spam { background: #fde8e8; border: 2px solid #d33; color: #a00; }
.ham { background: #e6f6ea; border: 2px solid #2a8a4a; color: #145c2a; }
.metrics { margin-top: 24px; background: #f7f7f7; border-radius: 10px; padding: 16px; }
.metrics h2 { margin-top: 0; font-size: 19px; }
.metrics table { border-collapse: collapse; }
.metrics td { padding: 5px 14px 5px 0; }
.small { color: #555; font-size: 14px; }
"""


def page(message="", result=None):
    """Build the HTML page. result = (label, spam_prob) or None."""
    result_html = ""
    if result is not None:
        label, spam_prob = result
        if label == "spam":
            result_html = (
                f'<div class="result spam" id="result">'
                f'&#9888; <b>SPAM</b><br>'
                f'<span class="small">Spam probability: {spam_prob:.0%}</span></div>')
        else:
            result_html = (
                f'<div class="result ham" id="result">'
                f'&#10003; <b>HAM</b> (normal message)<br>'
                f'<span class="small">Spam probability: {spam_prob:.0%}</span></div>')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMS Spam Detector</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>SMS Spam Detector</h1>
<p class="sub">ML Lab Mids Project &mdash; Naive Bayes (Lab 4) &middot; Jawad Mazhar, Iqra University</p>
<form method="post" action="/predict">
<textarea name="message" id="message" placeholder="Type a message here...">{html.escape(message)}</textarea>
<div class="row">
<button type="submit" id="predictBtn">Predict</button>
<a class="sample" href="/?sample=spam">Try a spam message</a>
<a class="sample" href="/?sample=ham">Try a normal message</a>
</div></form>
{result_html}
<div class="metrics" id="metrics">
<h2>Model Results (real test set)</h2>
<table>
<tr><td>Dataset</td><td><b>{TOTAL:,} messages</b> ({HAM:,} ham, {SPAM:,} spam)</td></tr>
<tr><td>Baseline (CountVectorizer + Naive Bayes)</td><td><b>98.74% accuracy</b></td></tr>
<tr><td>Improved (TF-IDF + cleaning + Naive Bayes)</td><td><b>{ACCURACY:.2%} accuracy</b></td></tr>
<tr><td>Cross-validation mean (5-fold)</td><td><b>95.15%</b></td></tr>
<tr><td>Confusion matrix</td><td>Ham: {CM[0][0]}, {CM[0][1]} &nbsp;|&nbsp; Spam: {CM[1][0]}, {CM[1][1]}</td></tr>
</table>
<p class="small">Source: UCI SMS Spam Collection. Improved accuracy is honestly
slightly below baseline on this test set &mdash; the improvement is in
cleaning, TF-IDF features, cross-validation and the live demo.</p>
</div></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, code=200):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        sample = query.get("sample", [""])[0]
        message = ""
        result = None
        if sample == "spam":
            message = SAMPLE_SPAM
            result = predict(message)
        elif sample == "ham":
            message = SAMPLE_HAM
            result = predict(message)
        self._send(page(message, result))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        fields = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
        message = fields.get("message", [""])[0]
        result = predict(message) if message.strip() else None
        self._send(page(message, result))

    def log_message(self, *args):
        pass  # keep the console clean


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"SMS Spam Detector app running at http://localhost:{PORT}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    server.serve_forever()
