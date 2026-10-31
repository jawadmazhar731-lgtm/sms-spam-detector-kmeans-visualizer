import os
from PIL import Image, ImageDraw, ImageFont
BASE=os.path.expanduser("~/workspace/mids-project/video")
S=os.path.join(BASE,"slides"); os.makedirs(S,exist_ok=True)
W,H=1280,720
BG=(16,48,36); BOARD=(22,62,46); WHITE=(250,250,240); YELLOW=(255,215,80); GREY=(190,200,190)
def font(s,b=True):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if b else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
def wrap(d,text,f,maxw):
    lines,cur=[], ""
    for w in text.split():
        t=(cur+" "+w).strip()
        if d.textlength(t,font=f)<=maxw: cur=t
        else: lines.append(cur); cur=w
    if cur: lines.append(cur)
    return lines
def board(title,subtitle,lines,fname):
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    d.rectangle([30,30,W-30,H-30],fill=BOARD,outline=(150,110,60),width=6)
    d.text((70,55),title,font=font(46),fill=YELLOW)
    if subtitle: d.text((70,125),subtitle,font=font(28,b=False),fill=GREY)
    y=195; f=font(32,b=False)
    for line in lines:
        for ln in wrap(d,line,f,1100):
            d.text((75,y),ln,font=f,fill=WHITE); y+=44
        y+=16
    img.save(f"{S}/{fname}"); print(fname)
def terminal(lines,fname):
    img=Image.new("RGB",(W,H),(10,10,10)); d=ImageDraw.Draw(img)
    d.text((40,30),"Real output - python3 sms_spam_detector.py",font=font(26,b=False),fill=(120,255,170))
    f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",26)
    y=90
    for ln in lines:
        d.text((40,y),ln,font=f,fill=WHITE); y+=38
    img.save(f"{S}/{fname}"); print(fname)

board("Mids Project - SMS Spam Detector","ML Lab, Lab 4: Naive Bayes  |  Student project",
 ["Ye project SMS parh kar batata hai:",
  "Spam hai ya Ham (asli message).",
  "Aaj main isko chala kar dikhata hoon,",
  "step by step, aaram se."],"slide1.png")
board("Project kya karta hai?","Ek idea, bas",
 ["Aap ek message likhte ho.",
  "Model jawab deta hai: Spam ya Ham.",
  "Bas. Itna hi kaam hai.",
  "Ab asli demo dekhte hain."],"slide2.png")
terminal([
 "--- LIVE DEMO ---",
 "",
 "Message: Congratulations! You have won",
 "a FREE prize, click this link now",
 "Prediction: spam",
 "",
 "Message: WINNER! You won 1000 cash,",
 "call now to collect your prize",
 "Prediction: spam",
 "",
 "Message: Hi, are we meeting for",
 "class tomorrow at 8:30?",
 "Prediction: ham",
 "",
 "Message: Ok, I will bring the",
 "lab manual tomorrow",
 "Prediction: ham",
],"slide3.png")
board("Data kitna tha?","Asli dataset, UCI SMS Spam Collection",
 ["Total messages: 5572",
  "Ham (asli): 4825",
  "Spam: 747",
  "80% se seekha, 20% se test kiya."],"slide4.png")
board("Step 1 - Cleaning","Ganda text saaf karo",
 ["1. Sab small letters me karo.",
  "2. Numbers aur punctuation hatao.",
  "3. Stopwords hatao: the, is, a.",
  "4. Stemming: winning -> winn.",
  "Example: FREE prize -> free prize."],"slide5.png")
board("Step 2 - TF-IDF","Ek line me",
 ["Important lafz ko zyada weight,",
  "aam lafz ko kam weight.",
  "Aur word pairs bhi seekhe,",
  "jaise 'free prize'."],"slide6.png")
board("Step 3 - Naive Bayes","Ek line me",
 ["Har lafz ki probability dekho,",
  "phir faisla karo: Spam ya Ham.",
  "'Naive' matlab: har lafz ko",
  "doosre se azad maan lo."],"slide7.png")
terminal([
 "--- REAL RESULTS ---",
 "",
 "BASELINE: Count + Naive Bayes",
 "Accuracy: 0.9874",
 "Confusion: ham [964, 2]",
 "           spam [12, 137]",
 "",
 "IMPROVED: TF-IDF + Cleaning",
 "Accuracy: 0.9839",
 "Confusion: ham [964, 2]",
 "           spam [16, 133]",
 "",
 "Cross-validation mean: 0.9515",
 "Farq sirf 4 test messages ka hai.",
],"slide8.png")
board("Viva me kya bolna hai?","Aur kaise chalana hai?",
 ["Naive Bayes: probability se faisla.",
  "TF-IDF: important lafz, zyada weight.",
  "Improvement: cleaning, TF-IDF,",
  "word pairs, cross-validation, live demo.",
  "Chalane ke liye: python3 sms_spam_detector.py",
  "Phir message likho, jawab milega."],"slide9.png")
print("slides done")
