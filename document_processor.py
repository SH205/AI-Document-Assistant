# This file will replace most of what ingest.py, chunking.py, and embeddings.py currently do.

# Import libraries
import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:

    def __init__(self):
        """
        Initialize the embedding model and text splitter.
        """

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Split large documents into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        

    def load_pdf(self, pdf_path):
        """
        Read one PDF and return the text from each page.
        """

        document = fitz.open(pdf_path)

        pages = []

        for page_num, page in enumerate(document):

            text = page.get_text()

            pages.append({
                "filename": Path(pdf_path).name,
                "page": page_num + 1,
                "content": text
            })

        return pages

    def create_chunks(self, pages):
        """
        Split each page into smaller chunks.
        """

        chunks = []

        for page in pages:

            split_text = self.splitter.split_text(page["content"])

            for chunk_num, chunk in enumerate(split_text):

                chunks.append({
                    "filename": page["filename"],
                    "page": page["page"],
                    "chunk_id": chunk_num,
                    "content": chunk

                })

        return chunks

    def create_embeddings(self, chunks):
        """
        Generate an embedding for every chunk.
        """

        for chunk in chunks:

            chunk["embedding"] = self.model.encode(
                chunk["content"]
            )

        return chunks