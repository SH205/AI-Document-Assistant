# Store embeddings in FAISS
# Import libraries
import faiss
import numpy as np
from pathlib import Path


class VectorStore:

    def __init__(self):
        """
        Initialize the vector store.
        """

        self.index = None
        self.chunks = None

    def build(self, chunks):
        """
        Build a FAISS index from document chunks.
        """

        # Save chunks for later retrieval
        self.chunks = chunks

        # Convert embeddings into a NumPy array
        vectors = np.array(
            [chunk["embedding"] for chunk in chunks],
            dtype="float32"
        )

        # Create FAISS index
        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        # Add vectors
        self.index.add(vectors)

    def save(self, folder="vector_store"):
        """
        Save the FAISS index.
        """

        Path(folder).mkdir(exist_ok=True)

        faiss.write_index(
            self.index,
            f"{folder}/index.faiss"
        )

    def load(self, folder="vector_store"):
        """
        Load a saved FAISS index.
        """

        self.index = faiss.read_index(
            f"{folder}/index.faiss"
        )

    def search(self, embedding, k=2):
        """
        Return the indices of the most relevant chunks.
        """

        distances, results = self.index.search(
            embedding,
            k
        )

        return results[0]

