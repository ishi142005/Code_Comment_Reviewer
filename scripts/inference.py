import joblib
from collections import Counter
import pandas as pd
import argparse

clf = joblib.load("models/linear_svc_code_comments.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_comment(comment):
    """Predict label for a single comment."""
    X_vec = vectorizer.transform([comment])
    return clf.predict(X_vec)[0]

def classify_comments(comments):
    """
    Classify a list of comments and return distribution counts + percentages.
    """
    labels = [predict_comment(c) for c in comments]
    counts = Counter(labels)
    total = sum(counts.values())
    percentages = {label: round((count/total)*100, 2) for label, count in counts.items()}

    df = pd.DataFrame({
        "Label": counts.keys(),
        "Count": counts.values(),
        "Percentage": [percentages[label] for label in counts.keys()]
    })
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Comment Classifier")
    parser.add_argument("--comment", type=str, help="A single comment to classify")
    parser.add_argument("--file", type=str, help="Path to a text file containing comments (one per line)")
    args = parser.parse_args()

    if args.comment:
        label = predict_comment(args.comment)
        print(f"Comment: {args.comment}\nPredicted label: {label}\n")
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                comments = [line.strip() for line in f if line.strip()]
            result_df = classify_comments(comments)
            print("\n--- Classification Summary ---")
            print(result_df.to_string(index=False))
        except FileNotFoundError:
            print(f"File not found: {args.file}")
    else:
        test_comments = [
            "This function sometimes returns None instead of raising an error.",
            "Not sure if this logic covers all edge cases. I think this might fail with empty lists, maybe handle it differently.",
            "Is this the intended behavior?",
            "This function crashes sometimes, maybe we should handle this differently.",
            "Can we rename this variable? It doesn’t break anything but is confusing.",
            "Edge case might fail, also could use some docstring improvements."
        ]
        result_df = classify_comments(test_comments)
        print("\n--- Classification Summary ---")
        print(result_df.to_string(index=False))
