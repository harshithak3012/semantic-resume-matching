import pandas as pd
from utils import clean_text

INPUT_PATH = "data/resumes.csv"
OUTPUT_PATH = "outputs/clean_resumes.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # Remove null / empty resumes
    df = df.dropna(subset=["Resume_str"])
    df = df[df["Resume_str"].str.strip() != ""]

    # Clean resume text
    df["clean_resume"] = df["Resume_str"].apply(clean_text)

    # add category context
    df["final_resume_text"] = df.apply(
        lambda row: f"""
        Professional Resume Profile
        Primary Role: {row['Category']}

        Skills, Experience, and Responsibilities:
        {row['clean_resume']}
        """,
            axis=1
        )


    # Keep only what we need
    final_df = df[["ID", "final_resume_text", "Category"]]
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Clean resumes saved to {OUTPUT_PATH}")
    print(f"Total resumes processed: {len(final_df)}")

if __name__ == "__main__":
    main()
