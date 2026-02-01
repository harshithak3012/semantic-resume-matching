import pandas as pd
from embedder import load_embedding_model
from endee import Endee

# Configuration
INDEX_NAME = "resumes_index"
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    job_text = """
    Job Title: Chef

Role Overview:
We are hiring a Chef to prepare high-quality meals and manage kitchen
operations.

Key Responsibilities:
- Prepare and cook dishes
- Maintain food quality and hygiene
- Plan menus
- Manage kitchen staff
- Control inventory and costs

Required Skills:
Cooking, Menu Planning, Food Safety, Time Management

Experience:
2–6 years of culinary experience

    """


    print("\nJob Description Used:\n")
    print(job_text[:500], "...\n")

    print("Loading embedding model...")
    model = load_embedding_model()
    

    # Embed job description
    print("Generating job embedding...")
    job_embedding = model.encode(
        job_text,
        normalize_embeddings=True
    ).tolist()

    # Connect to Endee
    endee = Endee()
    index = endee.get_index(INDEX_NAME)

    # Query Endee 
    print(f"Searching top {TOP_K} matching resumes...")
    results = index.query(
        job_embedding,
        TOP_K
    )

    # Display results
    print("\nTop Matching Resumes:\n")

    for rank, match in enumerate(results, start=1):
        resume_id = match.get("id", "N/A")
        metadata = match.get("metadata", {})

        # Handle Endee result format
        if "score" in match:
            similarity = match["score"]
        elif "similarity" in match:
            similarity = match["similarity"]
        elif "distance" in match:
            similarity = 1 - match["distance"]
        else:
            similarity = None

        category = metadata.get("category", "N/A")

        if similarity is not None:
            print(
                f"{rank}. Resume ID: {resume_id} | "
                f"Similarity: {similarity:.4f} | "
                f"Category: {category}"
            )
        else:
            print(
                f"{rank}. Resume ID: {resume_id} | "
                f"Category: {category}"
            )



if __name__ == "__main__":
    main()
