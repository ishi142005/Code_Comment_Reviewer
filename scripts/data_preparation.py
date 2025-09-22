import pandas as pd
import re
import os

def merge_text_files(file_paths, output_csv):
    """
    Reads multiple text files, merges them line by line into a single CSV.
    Each line is treated as a comment.
    """
    all_comments = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                all_comments.extend(lines)
        else:
            print(f"Warning: {file_path} not found.")

    df = pd.DataFrame({"Comment": all_comments})
    df.to_csv(output_csv, index=False)
    print(f"Merged {len(all_comments)} comments into {output_csv}")


def create_initial_dataset(raw_data_path, output_path):
    """
    Reads raw comment data, cleans it, applies heuristic labels,
    removes duplicates, and saves the final clean dataset.
    """
    df = pd.read_csv(raw_data_path)
    clean_comments = []

    for comment in df['Comment']:
        block = re.split(r"#Code:", comment, flags=re.IGNORECASE)[0]
        block = re.sub(r"#No.*|#File:.*", "", block)
        block = re.sub(r"/\*\*|\*/", "", block)
        block = re.sub(r"^\s*\*+", "", block, flags=re.MULTILINE)
        block = re.sub(r"@param\s+\w+|@return|@throws\s+\w+|@{?inheritDoc", "", block, flags=re.IGNORECASE)
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) > 5:
            clean_comments.append(block)

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
    df_clean = pd.DataFrame({"Comment": clean_comments, "Label": labels})
    df_clean = df_clean.drop_duplicates(subset="Comment").reset_index(drop=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Clean, labeled dataset saved to {output_path}")


if __name__ == "__main__":
    merge_text_files(
        file_paths=["data/coherent.txt", "data/noncoherent.txt"],
        output_csv="data/comments_for_labeling.csv"
    )

    create_initial_dataset(
        raw_data_path="data/comments_for_labeling.csv",
        output_path="data/labeled_comments_nodup.csv"
    )
