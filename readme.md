# Code Review Comment Classifier
## 🚀 Overview
This project is an end-to-end Natural Language Processing (NLP) pipeline designed to classify code review comments into four distinct, actionable categories: Bug, Question, Suggestion, and Style Nitpick.
The primary goal is not just to classify individual comments, but to provide a foundation for an analytics tool that can help engineering teams understand and improve their code review process. By analyzing feedback patterns at scale, teams can identify areas of the codebase with high bug rates, uncover gaps in documentation, or streamline their coding standards.
The project is built entirely in Python, leveraging core data science libraries like Pandas, NLTK, and Scikit-learn.

## 🛠️ Tech Stack
1. Language: Python 3.x
2. Data Science: Pandas, NLTK, Scikit-learn
3. Model & Artifacts: Joblib

## 🔬 Phase 1: Core Model Development (Completed)
This initial phase focused on building and validating the core machine learning engine. The workflow followed a structured, professional ML pipeline:
1. Data Ingestion & Cleaning: Raw comment data was ingested and sanitized using a robust Regex pipeline to remove code snippets, Javadoc tags (@param), and other non-textual noise.
2. Heuristic Labeling: An initial labeled dataset was created by applying a keyword-based heuristic, assigning labels based on trigger words (e.g., "error," "bug" -> Bug).
3. Robust Training Methodology: To prevent data leakage, the dataset was split into training and testing sets before any augmentation.
4. Data Augmentation (EDA): The class imbalance in the training set only was addressed using Easy Data Augmentation (EDA) techniques—specifically Synonym Replacement and Random Deletion—to create a balanced dataset for the model to learn from.
5. Model Training & Evaluation: A Linear Support Vector Classifier (LinearSVC) was trained on TF-IDF features extracted from the balanced training data. The model's performance was then validated against the untouched, held-out test set.

## 📊 Results & Performance
The model achieved a weighted accuracy of 93% on the unseen test data. The detailed performance metrics reveal a nuanced picture:

precision    recall  f1-score   support

          Bug       0.71      0.79      0.75        68
     Question       0.67      0.52      0.59        23
Style Nitpick       0.64      0.56      0.60        32
   Suggestion       0.96      0.96      0.96       908

     accuracy                           0.93      1031
    macro avg       0.75      0.71      0.72      1031
 weighted avg       0.93      0.93      0.93      1031

## Analysis of Results
The model demonstrates strong performance in identifying the majority class (Suggestion) and Bugs.
It shows a clear opportunity for improvement in distinguishing between Question and Style Nitpick comments. This is likely due to semantic overlap in language and some noise introduced by the initial heuristic labeling.

## 🔮 Phase 2: Project Vision & Future Work
1. The core model serves as a powerful proof-of-concept. The next phase focuses on evolving this from a script into a dynamic engineering analytics tool.
2. Live GitHub API Integration: Develop a module to connect directly to the GitHub API, allowing users to input a public repository URL and have the system automatically fetch and classify all pull request comments.
3. Interactive Dashboard: Build a lightweight web interface using Streamlit or Flask to display the results. The dashboard would feature visualizations showing the distribution of comment types, helping managers quickly identify trends.
4. Advanced NLP Models: To improve classification accuracy on nuanced comments, the next step would involve moving beyond TF-IDF to more sophisticated, context-aware embeddings like Sentence-BERT.
5. Improved Data Labeling: Transition from a purely heuristic system to a semi-supervised approach using a tool like Snorkel to programmatically generate a larger, more accurate training dataset.


## 🔧 How to Run the Core Model
Clone the repository:

```
git clone https://github.com/ishi142005/Code_Comment_Reviewer.git
cd Code_Comment_Reviewer
```

Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

```
pip install -r requirements.txt
```

Run the training pipeline:

```
python scripts/train_model.py
```

This will train the model and save the artifacts to the models/ directory.
Make a prediction on a new comment:

```
python scripts/inference.py --comment "This logic seems incorrect and might cause a bug.
```