import streamlit as st
import joblib
import pandas as pd
import requests
import re
import plotly.express as px  ### NEW: Import Plotly for better charts

# --- Page Configuration (Do this first) ---
st.set_page_config(
    page_title="Code Comment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading (No changes needed) ---
@st.cache_resource
def load_models():
    """Loads the trained model and vectorizer from disk."""
    try:
        clf = joblib.load("models/linear_svc_code_comments.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        return clf, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None

clf, vectorizer = load_models()

# --- API & Helper Functions (No changes needed) ---
def parse_pr_url(url):
    """Extracts owner, repo, and PR number from a GitHub PR URL."""
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", url)
    return match.groups() if match else (None, None, None)

def fetch_pr_comments_api(owner, repo_name, pr_number):
    """Fetch review comments from a PR using GitHub public API."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}/comments"
    headers = {"Accept": "application/vnd.github.v3+json"} # Best practice
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # This will raise an error for 4xx or 5xx status codes
        data = response.json()
        return [item['body'] for item in data]
    except requests.exceptions.HTTPError as e:
        st.error(f"Error fetching comments: {e.response.status_code}. Is the repo public and the URL correct?")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# ==============================================================================
#                          --- UI Application ---
# ==============================================================================

### UI Change 1: Use the sidebar for inputs to keep the main page clean.
with st.sidebar:
    st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=100)
    st.header("Analyze a Pull Request")
    pr_url = st.text_input(
        "Enter a public GitHub PR URL:", 
        placeholder="e.g., .../streamlit/streamlit/pull/1234"
    )
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    ### UI Change 2: Put the playground in the sidebar too. It's a secondary feature.
    with st.expander("🧪 Test a Single Comment"):
        single_comment = st.text_area("Enter comment:", height=100)
        if st.button("Classify", use_container_width=True):
            if single_comment and clf:
                X_single = vectorizer.transform([single_comment])
                prediction = clf.predict(X_single)[0]
                st.info(f"Predicted Label: **{prediction}**")
            elif not single_comment:
                st.warning("Please enter a comment.")


### UI Change 3: A more professional and welcoming main page.
st.title("🤖 Code Review Comment Analyzer")
st.markdown("""
    Welcome! This tool uses a trained NLP model to analyze code review comments from a public GitHub pull request. 
    It helps identify feedback patterns to improve engineering workflows. 
    
    **To get started, paste a public GitHub PR URL into the sidebar and click 'Analyze'.**
""")

st.markdown("---") # A nice visual separator

# --- Main Logic for Displaying Results (Only runs after button press) ---
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
                    
                    ### UI Change 4: Use metrics and a more visually appealing summary.
                    st.success(f"Analysis Complete! Found and classified **{len(comments)}** comments.")
                    
                    label_counts = pd.Series(predictions).value_counts()

                    # --- Display Key Metrics ---
                    cols = st.columns(len(label_counts))
                    for i, (label, count) in enumerate(label_counts.items()):
                        with cols[i]:
                            st.metric(label=label, value=count)

                    # --- Display a more beautiful Plotly chart ---
                    st.subheader("📊 Classification Breakdown")
                    fig = px.pie(
                        values=label_counts.values, 
                        names=label_counts.index, 
                        title='Distribution of Comment Types',
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    ### UI Change 5: Show the raw data in an expander, so it doesn't clutter the main view.
                    with st.expander("📄 View Raw Comments & Predictions"):
                        results_df = pd.DataFrame({'Comment': comments, 'Predicted Label': predictions})
                        st.dataframe(results_df)

elif analyze_button and not pr_url:
    st.warning("Please enter a URL in the sidebar to analyze.")