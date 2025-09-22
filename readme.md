# Code Review Comment Classifier

## 🚀 Overview

This project is an end-to-end Natural Language Processing (NLP) pipeline designed to classify code review comments from GitHub into four distinct, actionable categories: **Bug**, **Question**, **Suggestion**, and **Style Nitpick**.

The goal is to move beyond simple sentiment analysis and provide a more nuanced understanding of the feedback that developers give and receive. This can help engineering teams identify patterns in their code review process, such as a high frequency of "Bug" related comments, which might indicate a need for better pre-commit testing.

The project is built entirely in Python, leveraging core data science libraries like Pandas, NLTK, and Scikit-learn.


## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Core Libraries:**
  - `Pandas`: For data manipulation and management.
  - `NLTK`: For NLP tasks, specifically for data augmentation using its WordNet thesaurus.
  - `Scikit-learn`: For feature extraction (TF-IDF) and machine learning model training (LinearSVC).
- **Tooling:**
  - `Jupyter Notebooks`: For initial exploration and data cleaning.
  - `Joblib`: For serializing and saving the trained model and vectorizer.

## ⚙️ Project Pipeline

The project follows a classic, structured machine learning workflow:

1.  **Data Ingestion & Cleaning:**
    - Raw comment data was ingested from text files.
    - A robust cleaning pipeline using regular expressions (Regex) was created to remove code snippets, Javadoc tags (`@param`, `@return`), and other irrelevant artifacts, isolating the pure natural language of the comment.

2.  **Heuristic Labeling:**
    - To create an initial labeled dataset, a keyword-based heuristic was applied. Comments were automatically assigned a label based on the presence of specific trigger words (e.g., "error," "bug" -> Bug; "why," "how" -> Question).

3.  **Data Augmentation (EDA):**
    - An initial analysis revealed a severe class imbalance, with the 'Suggestion' class dominating the dataset.
    - To address this, Easy Data Augmentation (EDA) techniques—specifically Synonym Replacement and Random Deletion—were applied **only to the training set**. This was a critical step to prevent data leakage and ensure the test set remained representative of real-world, unseen data.

4.  **Model Training & Evaluation:**
    - The cleaned and augmented text data was converted into numerical features using a TF-IDF vectorizer.
    - A Linear Support Vector Classifier (LinearSVC) was trained on the balanced training data.
    - The model's performance was evaluated on the untouched, held-out test set.

## 📊 Results & Performance

The model achieved a **weighted accuracy of 93%** on the unseen test data. However, a closer look at the per-class metrics reveals a more nuanced story:
```
           precision    recall  f1-score   support

      Bug       0.71      0.79      0.75        68
 Question       0.67      0.52      0.59        23
Style Nitpick   0.64      0.56      0.60        32
Suggestion      0.96      0.96      0.96       908
 accuracy                           0.93      1031
macro avg       0.75      0.71      0.72      1031
weighted avg    0.93      0.93      0.93      1031
```


### Analysis of Results

-   The model performs exceptionally well on the majority class (`Suggestion`) and very well on identifying `Bugs`.
-   It shows weaker performance on distinguishing between `Question` and `Style Nitpick` comments. This is likely due to semantic overlap in the language used and potential noise introduced by the initial heuristic labeling. For example, a comment like "Could you fix this typo?" has the intent of a Style Nitpick but might be heuristically labeled as a Suggestion.


## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd code-comment-reviewer
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Run the training pipeline:**
    ```bash
    python scripts/train_model.py
    ```
    This will train the model and save the artifacts to the `models/` directory.

4.  **Make a prediction:**
    ```bash
    python scripts/inference.py --comment "This is a great suggestion, but I think you have a small bug here."
    ```