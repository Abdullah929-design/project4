import re, string, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import os

DATA_FAKE = "data/Fake.csv"
DATA_TRUE = "data/True.csv"
ARTIFACT_DIR = "models"

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

def main():
    df_fake = pd.read_csv(DATA_FAKE)
    df_true = pd.read_csv(DATA_TRUE)
    df_fake["class"] = 0
    df_true["class"] = 1

    df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df[[c for c in df.columns if c in ("text", "class")]]

    df["text"] = df["text"].astype(str).apply(wordopt)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["class"], test_size=0.25, random_state=42, stratify=df["class"]
    )

    vectorizer = TfidfVectorizer()
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test  = vectorizer.transform(X_test)

    LR = LogisticRegression(max_iter=1000)
    DT = DecisionTreeClassifier(random_state=42)
    GB = GradientBoostingClassifier(random_state=42)
    RF = RandomForestClassifier(random_state=42)

    LR.fit(Xv_train, y_train)
    DT.fit(Xv_train, y_train)
    GB.fit(Xv_train, y_train)
    RF.fit(Xv_train, y_train)

    print("LR test acc:", LR.score(Xv_test, y_test))
    print("DT test acc:", DT.score(Xv_test, y_test))
    print("GB test acc:", GB.score(Xv_test, y_test))
    print("RF test acc:", RF.score(Xv_test, y_test))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(vectorizer, f"{ARTIFACT_DIR}/vectorizer.joblib")
    joblib.dump(LR, f"{ARTIFACT_DIR}/logistic.joblib")
    joblib.dump(DT, f"{ARTIFACT_DIR}/decisiontree.joblib")
    joblib.dump(GB, f"{ARTIFACT_DIR}/gradientboost.joblib")
    joblib.dump(RF, f"{ARTIFACT_DIR}/randomforest.joblib")
    print("Saved all models to", ARTIFACT_DIR)

if __name__ == "__main__":
    main()
