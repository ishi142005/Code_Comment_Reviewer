import pandas as pd
import re

def create_initial_dataset(raw_data_path, output_path):
    """
    Reads raw comment data, cleans it, applies heuristic labels,
    removes duplicates, and saves the final clean dataset.
    """
    
    with open(raw_data_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"#Comment:", content, flags=re.IGNORECASE)

    def extract_clean_comment(block):
        try:
            comment = re.split(r"#Code:", block, flags=re.IGNORECASE)[0]
            comment = re.sub(r"#No.*|#File:.*", "", comment)
            comment = re.sub(r"/\*\*|\*/", "", comment)
            comment = re.sub(r"^\s*\*+", "", comment, flags=re.MULTILINE)
            comment = re.sub(r"@param\s+\w+|@return|@throws\s+\w+|@{?inheritDoc", "", comment, flags=re.IGNORECASE)
            comment = re.sub(r"\s+", " ", comment).strip()
            return comment if len(comment) > 5 else None
        except IndexError:
            return None

    clean_comments = [extract_clean_comment(b) for b in blocks]
    clean_comments = [c for c in clean_comments if c]

    def assign_label(comment):
        c = comment.lower()
        if any(word in c for word in ["error","fail","crash","wrong","bug","exception","incorrect"]):
            return "Bug"
        elif any(word in c for word in ["why","what","how","clarify","explain","question","unsure"]):
            return "Question"
        elif any(word in c for word in ["consider","should","maybe","could","suggestion","improve","alternative"]):
            return "Suggestion"
        elif any(word in c for word in ["format","naming","style","whitespace","indent","typo","convention"]):
            return "Style Nitpick"
        else:
            return "Suggestion" 

    labels = [assign_label(c) for c in clean_comments]

    df = pd.DataFrame({"Comment": clean_comments, "Label": labels})

    print(f"Before removing duplicates: {len(df)}")
    df = df.drop_duplicates(subset="Comment").reset_index(drop=True)
    print(f"After removing duplicates: {len(df)}")
    
    df.to_csv(output_path, index=False)
    print(f"Clean, labeled dataset saved to {output_path}")

if __name__ == "__main__":
    create_initial_dataset(
        raw_data_path="data/comments_for_labeling.csv", 
        output_path="data/labeled_comments_nodup.csv"
    )