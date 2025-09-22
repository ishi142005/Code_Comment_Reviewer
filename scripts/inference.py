import joblib
import argparse

clf = joblib.load("models/linear_svc_code_comments.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_comment(comment):
    X_vec = vectorizer.transform([comment])
    return clf.predict(X_vec)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Comment Classifier")
    parser.add_argument("--comment", type=str, help="A single comment to classify")
    args = parser.parse_args()

    if args.comment:
        label = predict_comment(args.comment)
        print(f"Comment: {args.comment}\nPredicted label: {label}\n")
    else:
        test_comments = [
            "This function sometimes returns None instead of raising an error.",
            "Not sure if this logic covers all edge cases. I think this might fail with empty lists, maybe handle it differently.",
            "Is this the intended behavior?",
            "This function crashes sometimes, maybe we should handle this differently.",
            "Can we rename this variable? It doesn’t break anything but is confusing.",
            "Edge case might fail, also could use some docstring improvements."
        ]
        
        for c in test_comments:
            label = predict_comment(c)
            print(f"Comment: {c}\nPredicted label: {label}\n")
