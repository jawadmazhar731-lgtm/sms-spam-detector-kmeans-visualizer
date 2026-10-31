# Capture real screenshots + video frames of the running SMS Spam Detector app.
# Uses Playwright + headless Chromium against http://localhost:8765 (app.py).
# Every prediction shown comes from the trained model in the live app.
import os, sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8765"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app-screenshots")
FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video-frames")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

SPAM = "Congratulations! You have won a FREE prize, click this link now to claim"
HAM = "Hi, are we meeting for class tomorrow at 8:30?"

def shot(page, name):
    page.screenshot(path=os.path.join(OUT, name))

def frame(page, name):
    page.screenshot(path=os.path.join(FRAMES, name))

import os
CHROME = os.path.expanduser("~/workspace/.browsers/chromium-1248/chrome-linux/chrome")
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME,
                                args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    # --- Screenshots ---
    page.goto(BASE + "/")
    page.wait_for_selector("h1")
    shot(page, "01-home.png")
    page.goto(BASE + "/?sample=spam")
    page.wait_for_selector("#result")
    shot(page, "02-spam-result.png")
    page.goto(BASE + "/?sample=ham")
    page.wait_for_selector("#result")
    shot(page, "03-ham-result.png")
    page.goto(BASE + "/")
    page.evaluate("document.getElementById('metrics').scrollIntoView()")
    page.wait_for_timeout(400)
    shot(page, "04-metrics.png")
    # --- Video frames: app in use, step by step ---
    page.goto(BASE + "/")
    page.wait_for_selector("#message")
    frame(page, "f01-home.png")
    # Type the spam message in stages (looks like real typing)
    for i, cut in enumerate([12, 30, 50, len(SPAM)]):
        page.fill("#message", SPAM[:cut])
        frame(page, f"f02-type-spam-{i}.png")
    page.click("#predictBtn")
    page.wait_for_selector("#result")
    frame(page, "f03-spam-result.png")
    frame(page, "f04-spam-result.png")
    # Clear and type the ham message
    page.fill("#message", "")
    for i, cut in enumerate([12, 28, len(HAM)]):
        page.fill("#message", HAM[:cut])
        frame(page, f"f05-type-ham-{i}.png")
    page.click("#predictBtn")
    page.wait_for_selector("#result")
    frame(page, "f06-ham-result.png")
    frame(page, "f07-ham-result.png")
    page.evaluate("document.getElementById('metrics').scrollIntoView()")
    page.wait_for_timeout(400)
    frame(page, "f08-metrics.png")
    browser.close()
print("screenshots:", sorted(os.listdir(OUT)))
print("frames:", sorted(os.listdir(FRAMES)))
