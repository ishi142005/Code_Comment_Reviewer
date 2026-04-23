import joblib

def load_models():
    clf = joblib.load("models/linear_svc_code_comments.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    
    return clf, vectorizer