import re
import requests
import os


def parse_pr_url(url):
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", url)
    return match.groups() if match else (None, None, None)


def fetch_pr_comments(owner, repo, pr_number):
    token = os.getenv("GITHUB_TOKEN")

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    if token:
        headers["Authorization"] = f"token {token}"

    comments = []

    # Review comments
    review_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    review_res = requests.get(review_url, headers=headers)

    if review_res.status_code != 200:
        raise Exception(f"GitHub API Error: {review_res.status_code}")

    comments.extend([item["body"] for item in review_res.json()])

    # Issue comments
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    issue_res = requests.get(issue_url, headers=headers)

    if issue_res.status_code == 200:
        comments.extend([item["body"] for item in issue_res.json()])

    if not comments:
        raise Exception("No comments found on this PR")

    return comments