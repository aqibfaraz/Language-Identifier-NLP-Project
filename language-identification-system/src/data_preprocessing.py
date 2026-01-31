def preprocess(text):
    """
    Preprocess the input text by normalizing whitespace.
    For language identification, we keep case and punctuation as they are important features.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The cleaned text.
    """
    # Normalize whitespace only (keep case and punctuation for language detection)
    text = ' '.join(text.split())
    
    return text