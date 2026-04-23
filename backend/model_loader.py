import os
import joblib

def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    clf_path = os.path.join(BASE_DIR, "models", "linear_svc_code_comments.pkl")
    vec_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

    clf = joblib.load(clf_path)
    vectorizer = joblib.load(vec_path)

    return clf, vectorizer