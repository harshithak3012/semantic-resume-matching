import numpy as np
import pandas as pd
from endee import Endee

# Configuration
INDEX_NAME = "resumes_index"
EMBEDDINGS_PATH = "embeddings/resume_embeddings.npy"
METADATA_PATH = "embeddings/resume_metadata.csv"
BATCH_SIZE = 100


def main():
    print("Loading embeddings and metadata...")

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)

    dim = embeddings.shape[1]

    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimension: {dim}")

    # Initialize Endee (LOCAL OSS)
    endee = Endee()
    print("Connected to local Endee server (localhost:8080)")

    print("Checking for existing index...")
    indexes = endee.list_indexes() 

    if INDEX_NAME in indexes:
        print(f"Deleting existing index: {INDEX_NAME}")
        endee.delete_index(INDEX_NAME)

    
    # Create index 
    print(f"Creating index: {INDEX_NAME}")

    try:
        endee.create_index(
            INDEX_NAME,
            dim,
            "cosine"
        )
        print(f"Index '{INDEX_NAME}' created successfully")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Index '{INDEX_NAME}' already exists — continuing")
        else:
            raise e


    # Get index handle
    index = endee.get_index(INDEX_NAME)

    # Prepare vectors
    print("Preparing vectors for upload...")

    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(metadata_df.iloc[i]["resume_id"]),
            "vector": emb.tolist(),  
            "metadata": {
                "resume_id": str(metadata_df.iloc[i]["resume_id"]),
                "category": str(metadata_df.iloc[i]["category"]),
                "source": "kaggle_resume_dataset"
            }
        })

    # Upload vectors in batches
    print("Uploading resume embeddings to Endee...")

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(batch)
        print(f"   Uploaded {i + len(batch)} / {len(vectors)}")

    print("Resume embeddings successfully stored in Endee!")


if __name__ == "__main__":
    main()
