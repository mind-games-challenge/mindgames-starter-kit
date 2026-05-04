from .embed_openai import EmbeddingTransformer
from sentence_transformers import SentenceTransformer

OPENAI_PREFIX = 'openai/'

def get_embedding_model(model_name_or_path: str,) -> EmbeddingTransformer | SentenceTransformer:
    """
    Get the embedding client based on the model name. Retrieves an OpenAI client if the model name starts with
    ``OPENAI_PREFIX``.

    Args:
        model_name_or_path (str): the model name.

    Returns:
        EmbeddingTransformer | SentenceTransformer: the embedding client.
    """
    if model_name_or_path.startswith(OPENAI_PREFIX):  # OpenAI
        return EmbeddingTransformer(model_name_or_path.removeprefix(OPENAI_PREFIX))

    else:  # Hugging Face
        return SentenceTransformer(
            model_name_or_path=model_name_or_path,
            model_kwargs={"device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )