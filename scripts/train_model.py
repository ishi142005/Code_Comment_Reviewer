import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import random
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

random.seed(42)  

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(words, n=1):
    new_words = words.copy()
    eligible_words = [w for w in words if get_synonyms(w)]
    random.shuffle(eligible_words)
    num_replaced = 0
    for w in eligible_words:
        synonyms = get_synonyms(w)
        if synonyms:
            new_words = [random.choice(synonyms) if word==w else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def eda(sentence, alpha_sr=0.1, p_rd=0.1):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    
    # --- Synonym Replacement ---
    n_sr = max(1, int(alpha_sr * len(words)))
    words = synonym_replacement(words, n_sr)
    
    # --- Random Deletion ---
    if p_rd > 0:
        new_words = [w for w in words if random.random() > p_rd]
        if new_words:
            words = new_words
    
    return ' '.join(words)

def augment_and_balance_training_data(X_train, y_train):
    """
    Applies EDA augmentation ONLY to the training data to balance classes.
    """
    df_train = pd.DataFrame({'Comment': X_train, 'Label': y_train})
    target_count = df_train['Label'].value_counts().max()
    
    df_balanced_list = []
    
    for label in df_train['Label'].unique():
        subset = df_train[df_train['Label'] == label]
        df_balanced_list.append(subset)  # Keep original
        
        current_count = len(subset)
        n_needed = target_count - current_count
        
        if n_needed > 0:
            augmented_comments = []
            for _ in range(n_needed):
                original_comment = subset.sample(1)['Comment'].iloc[0]
                augmented_comments.append(eda(original_comment))
            
            df_aug = pd.DataFrame({'Comment': augmented_comments, 'Label': [label]*n_needed})
            df_balanced_list.append(df_aug)
    
    df_train_balanced = pd.concat(df_balanced_list).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_train_balanced['Comment'], df_train_balanced['Label']

# --- Main Training Function ---
def train_and_evaluate(csv_path="data/labeled_comments_nodup.csv"):
    # Load original dataset
    df_original = pd.read_csv(csv_path)
    df_original.dropna(subset=['Comment'], inplace=True)

    X = df_original['Comment']
    y = df_original['Label']

    # Split BEFORE augmentation (sanctuary test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Original split -> Train: {len(X_train)}, Test: {len(X_test)}")
    print("\nClass distribution in the UNTOUCHED test set:\n", y_test.value_counts())

    # Augment ONLY training data
    X_train_balanced, y_train_balanced = augment_and_balance_training_data(X_train, y_train)
    print(f"\nAfter augmentation -> Balanced Train: {len(X_train_balanced)}")
    print("\nClass distribution in the BALANCED training set:\n", y_train_balanced.value_counts())

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train_balanced)
    
    # Model with balanced class weights
    clf = LinearSVC(max_iter=2000, class_weight='balanced')

    # --- NEW: Cross-validation on TRAINING data ---
    print("\nRunning 5-Fold Cross-Validation on balanced training data...")
    cv_scores = cross_val_score(clf, X_train_vec, y_train_balanced, cv=5, scoring='accuracy')
    print("CV Scores:", cv_scores)
    print("Average CV accuracy:", cv_scores.mean())
    print("CV accuracy std deviation:", cv_scores.std())

    # Fit on full balanced training data
    clf.fit(X_train_vec, y_train_balanced)

    # Evaluate on untouched test set
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print("\n--- FINAL EVALUATION ON UNSEEN TEST DATA ---")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    joblib.dump(clf, "models/linear_svc_code_comments.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("\nCorrected model and vectorizer saved to 'models/'")

if __name__ == "__main__":
    train_and_evaluate()
