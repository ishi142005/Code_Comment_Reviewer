def predict_comments(comments, clf, vectorizer):
    X = vectorizer.transform(comments)
    predictions = clf.predict(X)
    return predictions