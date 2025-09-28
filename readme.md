# 🤖 Code Review Comment Classifier
[Live Application Link] 
## 🚀 Overview
This project is an end-to-end Natural Language Processing (NLP) pipeline designed to classify code review comments into four distinct, actionable categories: Bug, Question, Suggestion, and Style Nitpick.
The primary goal is not just to classify individual comments, but to provide a foundation for an engineering analytics tool. By analyzing feedback patterns at scale, teams can identify areas of the codebase with high bug rates, uncover gaps in documentation, or streamline their coding standards.
This project is built entirely in Python, leveraging core data science libraries like Pandas, NLTK, and Scikit-learn.
## 🛠️ Tech Stack

- Language: Python 3.x
- Data Science: Pandas, NLTK, Scikit-learn
- Web Framework: Streamlit
- Model & Artifacts: Joblib

## 🔬 Phase 1: Core Model Development (Completed)

This initial phase focused on building and validating the core machine learning engine. The workflow followed a structured, professional ML pipeline:

- **Data Ingestion & Cleaning**: Raw comment data was ingested and sanitized using a robust Regex pipeline to remove code snippets, Javadoc tags (@param), and other non-textual noise.
- **Heuristic Labeling**: An initial labeled dataset was created by applying a keyword-based heuristic to assign labels based on trigger words (e.g., "error," "bug" -> Bug).
- **Robust Training Methodology**: To prevent data leakage, the dataset was split into training and testing sets before any augmentation.
- **Data Augmentation (EDA)**: The class imbalance in the training set only was addressed using Easy Data Augmentation (EDA) techniques to create a balanced dataset for the model to learn from.
- **Model Training & Evaluation**: A Linear Support Vector Classifier (LinearSVC) was trained on TF-IDF features and validated against the untouched, held-out test set.

## 📊 Performance Metrics

- The model achieved a weighted accuracy of 93% on the unseen test data.
```
precision    recall  f1-score   support

          Bug       0.71      0.79      0.75        68
     Question       0.67      0.52      0.59        23
Style Nitpick       0.64      0.56      0.60        32
   Suggestion       0.96      0.96      0.96       908

     accuracy                           0.93      1031
    macro avg       0.75      0.71      0.72      1031
 weighted avg       0.93      0.93      0.93      1031
```

### Analysis of Results
- The model demonstrates strong performance in identifying the majority class (Suggestion) and Bugs.
- It shows a clear opportunity for improvement in distinguishing between Question and Style Nitpick comments, likely due to semantic overlap and some noise from the initial heuristic labeling.

## 🔮 Phase 2: Project Vision 
The core model serves as a powerful proof-of-concept. The next phase focuses on evolving this from a script into a dynamic analytics tool.
- **Live GitHub API Integration**: Connect to the GitHub API to allow users to analyze public pull requests in real-time.
- **Interactive Dashboard**: The current Streamlit app serves as the foundation. Future work would involve adding more advanced visualizations to track trends over time.
- **Advanced NLP Models**: To improve accuracy, the next step would involve moving beyond TF-IDF to more sophisticated, context-aware embeddings like Sentence-BERT.
- **Improved Data Labeling**: Transition from the heuristic system to a semi-supervised approach using a tool like Snorkel to programmatically generate a larger, more accurate training dataset.

## 🔧 How to Run Locally
- Clone the repository:
```
Bash
git clone https://github.com/ishi142005/Code_Comment_Reviewer.git
cd Code_Comment_Reviewer
```

- Create a virtual environment and install dependencies:
```
Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

- Run the Streamlit application:
```
Bash
streamlit run app.py
```