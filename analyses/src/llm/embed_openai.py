import numpy as np
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingTransformer:
    def __init__(self, model_name_or_path: str,):
        self.model_name_or_path = model_name_or_path
        load_dotenv()
        self.client = OpenAI()
        self.similarity_fn_name = "cosine similarity"

    def encode(self, text: List[str], batch_size: int) -> np.array:
        """
        Encodes ``text`` into separate embeddings. Uses the default OpenAI tokenizer.

        Args:
            text (List[str]): the texts to embed.
            batch_size (int): the embedding batch size.

        Returns:
            np.array: a matrix of embeddings.
        """
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]
            response = self.client.embeddings.create(input=batch,model=self.model_name_or_path,)
            embeddings.extend([r.embedding for r in response.data])
        return np.array(embeddings)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the row-wise similarity between two matrices.

        Args:
            a (np.ndarray): a matrix of shape (x, y).
            b (np.ndarray): a matrix of shape (x, y).

        Returns:
            np.array: a vector of similarities between ``a`` and ``b``. Has shape (x,).
        """
        assert a.shape == b.shape, f"Input matrices should have the same shape. Received {a.shape} and {b.shape}."
        if self.similarity_fn_name == "cosine similarity":
            return self._matrix_cosine(a, b)
        else:
            raise NotImplementedError(f"This module only supports 'cosine similarity',"
                                      f" received {self.similarity_fn_name}")

    @staticmethod
    def _matrix_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the cartesian row-wise similarity between two matrices.

        Args:
            a (np.ndarray): a matrix of shape (x, y).
            b (np.ndarray): a matrix of shape (x, y).

        Returns:
            np.array: a vector of cosine similarities between ``a`` and ``b``. Has shape (x,).
        """
        assert a.shape == b.shape, f"Input matrices should have the same shape. Received {a.shape} and {b.shape}."
        return cosine_similarity(a,b)
