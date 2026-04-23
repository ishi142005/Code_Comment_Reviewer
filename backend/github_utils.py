import re
import requests


def parse_pr_url(url):
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", url)
    return match.groups() if match else (None, None, None)


def fetch_pr_comments(owner, repo, pr_number):
    headers = {"Accept": "application/vnd.github.v3+json"}

    # Try review comments
    review_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    review_res = requests.get(review_url, headers=headers)

    comments = []

    if review_res.status_code == 200:
        review_data = review_res.json()
        comments.extend([item["body"] for item in review_data])

    # Try issue comments (discussion)
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    issue_res = requests.get(issue_url, headers=headers)

    if issue_res.status_code == 200:
        issue_data = issue_res.json()
        comments.extend([item["body"] for item in issue_data])

    if not comments:
        raise Exception("No comments found on this PR")

    return comments