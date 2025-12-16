import faiss
import pickle
from typing import List

class FAISSVectorStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, embedding: List[float], top_k: int):
        scores, indices = self.index.search(
            np.array([embedding]).astype("float32"),
            top_k
        )
        return scores[0], indices[0]
