import os, re, string, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- config ---
DEFAULT_FINAL_MODE = "LR"   # "LR" = Logistic Regression only, "VOTE" = majority of all 4

# --- app ---
app = Flask(__name__)

# allow override via env var, comma-separated list, default wildcard
_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()] or ["*"]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- load artifacts (match YOUR file names) ---
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
LR  = joblib.load(os.path.join(MODEL_DIR, "logistic.joblib"))
DT  = joblib.load(os.path.join(MODEL_DIR, "decisiontree.joblib"))
GB  = joblib.load(os.path.join(MODEL_DIR, "gradientboost.joblib"))
RF  = joblib.load(os.path.join(MODEL_DIR, "randomforest.joblib"))

def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def label(n: int) -> str:
    return "Fake News" if n == 0 else "Not A Fake News"

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    # accept either "news" or "text" field
    news = (data.get("news") or data.get("text") or "").strip()
    mode = (data.get("mode") or DEFAULT_FINAL_MODE).upper()

    if not news:
        return jsonify({"error": "Provide 'news' (or 'text') in JSON body"}), 400

    processed = wordopt(news)
    X = vectorizer.transform([processed])

    pred_LR = int(LR.predict(X)[0])
    pred_DT = int(DT.predict(X)[0])
    pred_GB = int(GB.predict(X)[0])
    pred_RF = int(RF.predict(X)[0])

    if mode == "VOTE":
        votes = [pred_LR, pred_DT, pred_GB, pred_RF]
        final = max(set(votes), key=votes.count)
    else:
        final = pred_LR  # your preference: LR decides final

    try:
        proba_LR = float(LR.predict_proba(X)[0][1])  # prob of "Not A Fake News"
    except Exception:
        proba_LR = None

    return jsonify({
        "mode": mode,
        "models": {
            "LR": label(pred_LR),
            "DT": label(pred_DT),
            "GB": label(pred_GB),
            "RF": label(pred_RF),
        },
        "LR_probability_real": proba_LR,
        "final": label(final)
    })

def _origin_allowed(origin: str) -> bool:
    if not origin:
        return False
    if "*" in ALLOWED_ORIGINS:
        return True
    return origin in ALLOWED_ORIGINS

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if _origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin if "*" not in ALLOWED_ORIGINS else "*"
    elif "*" in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Vary"] = "Origin"
    return response

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
