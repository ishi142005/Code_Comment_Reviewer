from fastapi import FastAPI
from pydantic import BaseModel

from model_loader import load_models
from github_utils import parse_pr_url, fetch_pr_comments
from predictor import predict_comments
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = []

frontend_url = os.getenv("FRONTEND_URL")
local_url = os.getenv("LOCAL_FRONTEND_URL")

if frontend_url:
    origins.append(frontend_url)

if local_url:
    origins.append(local_url)

# Fallback
if not origins:
    origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
clf, vectorizer = load_models()

class PRRequest(BaseModel):
    pr_url: str


@app.get("/")
def home():
    return {"message": "Code Comment Analyzer API is running"}


@app.post("/analyze")
def analyze_pr(request: PRRequest):
    # parse URL
    owner, repo, pr_number = parse_pr_url(request.pr_url)

    if not owner:
        return {"error": "Invalid GitHub PR URL"}

    try:
        # fetch comments
        comments = fetch_pr_comments(owner, repo, int(pr_number))

        if not comments:
            return {
                "message": "No comments found",
                "total_comments": 0,
                "label_counts": {},
                "results": []
            }

        # predict
        predictions = predict_comments(comments, clf, vectorizer)

        # format response
        results = []
        label_counts = {}

        for comment, label in zip(comments, predictions):
            results.append({
                "comment": comment,
                "label": label
            })

            label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "total_comments": len(comments),
            "label_counts": label_counts,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}