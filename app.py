import streamlit as st
import joblib
import pandas as pd
import requests
import re
import plotly.express as px
import os

st.set_page_config(
    page_title="Code Comment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_URL = "https://drive.google.com/drive/u/0/folders/18cvokJhn5BhY00HahNAdoyBgvth-Z7Sh?ths=true"
VECTORIZER_URL = "https://drive.google.com/drive/u/0/folders/18cvokJhn5BhY00HahNAdoyBgvth-Z7Sh?ths=true"

@st.cache_resource
def load_models():
    """Downloads and loads the trained model + vectorizer from Google Drive."""
    try:
        os.makedirs("models", exist_ok=True)

        model_path = "models/linear_svc_code_comments.pkl"
        if not os.path.exists(model_path):
            st.info("Downloading model file...")
            r = requests.get(MODEL_URL)
            with open(model_path, "wb") as f:
                f.write(r.content)

        vectorizer_path = "models/tfidf_vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            st.info("Downloading vectorizer file...")
            r = requests.get(VECTORIZER_URL)
            with open(vectorizer_path, "wb") as f:
                f.write(r.content)

        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return clf, vectorizer

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


clf, vectorizer = load_models()

def parse_pr_url(url):
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", url)
    return match.groups() if match else (None, None, None)

def fetch_pr_comments_api(owner, repo_name, pr_number):
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}/comments"
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item['body'] for item in data]
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching comments: {e.response.status_code}. Is the repo public and the URL correct?")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

with st.sidebar:
    st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
    st.header("Analyze a Pull Request")
    pr_url = st.text_input(
        "Enter a public GitHub PR URL:", 
        placeholder="e.g., .../streamlit/streamlit/pull/1234"
    )
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    with st.expander("🧪 Test a Single Comment"):
        single_comment = st.text_area("Enter comment:", height=100)
        if st.button("Classify", use_container_width=True):
            if single_comment and clf:
                X_single = vectorizer.transform([single_comment])
                prediction = clf.predict(X_single)[0]
                st.info(f"Predicted Label: **{prediction}**")
            elif not single_comment:
                st.warning("Please enter a comment.")

st.title("🤖 Code Review Comment Analyzer")
st.markdown("""
    Welcome! This tool uses a trained NLP model to analyze code review comments from a public GitHub pull request. 
    It helps identify feedback patterns to improve engineering workflows. 
    
    **To get started, paste a public GitHub PR URL into the sidebar and click 'Analyze'.**
""")
st.markdown("---")

if analyze_button and pr_url:
    if clf is None:
        st.error("Model is not loaded. Cannot proceed.")
        st.stop()

    owner, repo_name, pr_number = parse_pr_url(pr_url)
    
    if not owner:
        st.error("Invalid GitHub Pull Request URL. Please use the format: https://github.com/owner/repo/pull/number")
    else:
        with st.spinner(f"Fetching and analyzing comments for PR #{pr_number}... This may take a moment."):
            comments = fetch_pr_comments_api(owner, repo_name, int(pr_number))
            
            if comments is not None:
                if not comments:
                    st.warning("No review comments were found on this Pull Request. This can also happen if the PR is very old.")
                else:
                    X_new = vectorizer.transform(comments)
                    predictions = clf.predict(X_new)
                    
                    st.success(f"Analysis Complete! Found and classified **{len(comments)}** comments.")
                    
                    label_counts = pd.Series(predictions).value_counts()

                    cols = st.columns(len(label_counts))
                    for i, (label, count) in enumerate(label_counts.items()):
                        with cols[i]:
                            st.metric(label=label, value=count)

                    st.subheader("📊 Classification Breakdown")
                    fig = px.pie(
                        values=label_counts.values, 
                        names=label_counts.index, 
                        title='Distribution of Comment Types',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📄 View Raw Comments & Predictions"):
                        results_df = pd.DataFrame({'Comment': comments, 'Predicted Label': predictions})
                        st.dataframe(results_df)

elif analyze_button and not pr_url:
    st.warning("Please enter a URL in the sidebar to analyze.")
