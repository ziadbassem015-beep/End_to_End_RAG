# conftest.py — shared pytest configuration
# Adds the project root to sys.path so tests can import rag_system.*

import sys
from pathlib import Path
from typing import Any
import numpy as np

# Add the End_to_End_Agentic_RAG directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define Mock classes
class MockSentenceTransformer:
    def __init__(self, model_name: str = None, **kwargs: Any) -> None:
        self.model_name = model_name

    def encode(self, texts: Any, **kwargs: Any) -> Any:
        if isinstance(texts, str):
            input_list = [texts]
            is_single = True
        else:
            input_list = list(texts)
            is_single = False

        embeddings = []
        for text in input_list:
            # Deterministic pseudo-random generation based on text hash
            val = sum(ord(c) for c in text) % 256
            rng = np.random.default_rng(seed=val)
            vec = rng.random(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        arr = np.vstack(embeddings)
        
        convert_to_numpy = kwargs.get("convert_to_numpy", True)
        if is_single and isinstance(texts, str):
            return arr[0] if convert_to_numpy else arr[0].tolist()
        return arr if convert_to_numpy else arr.tolist()


class MockCrossEncoder:
    def __init__(self, model_name: str = None, **kwargs: Any) -> None:
        self.model_name = model_name

    def predict(self, pairs: Any, **kwargs: Any) -> Any:
        # Return mock scores based on text features so they look like cross-encoder scores
        scores = []
        for pair in pairs:
            # deterministic mock score
            val = sum(ord(c) for c in pair[1]) % 20 - 10
            scores.append(float(val))
        return np.array(scores, dtype=np.float32)


# Apply mock globally to sentence_transformers module attributes
import sentence_transformers
sentence_transformers.SentenceTransformer = MockSentenceTransformer
sentence_transformers.CrossEncoder = MockCrossEncoder
