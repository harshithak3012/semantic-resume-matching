def clean_text(text: str) -> str:
    """
    Minimal cleaning for embedding-based NLP tasks
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())
    return text.lower()
