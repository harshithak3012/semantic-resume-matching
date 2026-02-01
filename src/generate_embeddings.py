import os
import pandas as pd
import numpy as np
from embedder import load_embedding_model, generate_embeddings

INPUT_PATH = "outputs/clean_resumes.csv"
EMBEDDINGS_DIR = "embeddings"

EMBEDDING_OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, "resume_embeddings.npy")
METADATA_OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, "resume_metadata.csv")


def main():
    print("Loading cleaned resumes...")
    df = pd.read_csv(INPUT_PATH)

    texts = df["final_resume_text"].astype(str).tolist()
    resume_ids = df["ID"].astype(str).tolist()
    categories = df["Category"].astype(str).tolist()

    print(f"Total resumes to embed: {len(texts)}")

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(
        model,
        texts,
        normalize=True   
    )

    print("Saving embeddings and metadata...")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    np.save(EMBEDDING_OUTPUT_PATH, embeddings)

    metadata_df = pd.DataFrame({
        "resume_id": resume_ids,
        "category": categories
    })
    metadata_df.to_csv(METADATA_OUTPUT_PATH, index=False)

    print("Resume embeddings generated successfully!")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings → {EMBEDDING_OUTPUT_PATH}")
    print(f"Saved metadata   → {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
