# 🤖 Code Review Comment Analyzer

🔗 Live App: https://code-comment-reviewer-kx5pfmcjc-ishita-kanaujias-projects.vercel.app/

---

## 🚀 Overview

This project is a full-stack NLP-powered web application that analyzes GitHub pull request comments and classifies them into actionable categories:

- 🐞 Bug  
- ❓ Question  
- 💡 Suggestion  
- 🎨 Style Nitpick  

It helps developers and teams understand feedback patterns and improve code quality.

---

## 🏗️ Architecture

This project follows a **modern full-stack architecture**:
```
React (Frontend) → FastAPI (Backend) → ML Model (Scikit-learn)
```


- **Frontend (React + Vite)**  
  - User inputs GitHub PR URL  
  - Displays summary + categorized comments  

- **Backend (FastAPI)**  
  - Fetches PR comments using GitHub API  
  - Runs ML model for classification  
  - Returns structured results  

- **ML Model**  
  - TF-IDF + LinearSVC  
  - Trained on labeled code review comments  

---

## 🛠️ Tech Stack

### Frontend
- React (Vite)
- JavaScript
- CSS

### Backend
- FastAPI
- Python

### Machine Learning
- Scikit-learn
- Pandas
- NLTK
- Joblib

---

## 📊 Model Performance

- Accuracy: **93%**
```
precision recall f1-score support

Bug 0.71 0.79 0.75 68
Question 0.67 0.52 0.59 23
Style 0.64 0.56 0.60 32
Suggestion 0.96 0.96 0.96 908

accuracy 0.93 1031
```

---

## ⚙️ Features

- 🔗 Analyze real GitHub PRs
- 📊 Summary of comment categories
- 🧠 ML-based classification
- ⚡ Fast API responses
- 🎯 Clean and responsive UI

---

## 🧪 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/code-comment-reviewer.git
cd code-comment-reviewer
```

## Run Backend
```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run Frontend
```
cd frontend
npm install
npm run dev
```
## Create Environment Variables
### Create a .env file inside frontend/:
```
VITE_API_URL=http://127.0.0.1:8000
```

## Key Learnings
- Built full-stack app (React + FastAPI)
- Integrated ML model into production API
- Handled API communication & environment configs
- Designed clean UI/UX for real users