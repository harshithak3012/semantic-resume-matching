from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Loads and returns the sentence embedding model.
    Same model MUST be used for resumes and jobs.
    """
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(model, texts, batch_size=32, normalize=True):
    """
    Converts a list of texts into embeddings.
    Normalization is important for cosine similarity search.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # shape will be (num_texts, embedding_dim) for MiniLM models 384
    return embeddings
