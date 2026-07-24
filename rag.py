import ollama
from sentence_transformers import SentenceTransformer

from document_processor import DocumentProcessor
from vector_store import VectorStore
import re


class RAGPipeline:

    def __init__(self):

        # PDF processing
        self.processor = DocumentProcessor()

        # FAISS vector database
        self.store = VectorStore()

        # Embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Store all document chunks
        self.chunks = []


    def add_pdf(self, pdf_path):
        """
        Add a PDF to the existing knowledge base.
        """

        # Extract PDF text
        pages = self.processor.load_pdf(pdf_path)

        # Split into chunks
        new_chunks = self.processor.create_chunks(
            pages
        )

        # Create embeddings
        new_chunks = self.processor.create_embeddings(
            new_chunks
        )

        # Add new chunks to existing chunks
        self.chunks.extend(new_chunks)


    def build_index(self):
        """
        Create FAISS index after all PDFs are uploaded.
        """

        self.store.build(
            self.chunks
        )

        self.store.save()


    def ask(self, question):
        """
        Answer questions using all uploaded documents.
        """

        # Embed question
        question_embedding = self.model.encode(
            [question]
        )


        # Search vector database
        
        # FAISS semantic search
        semantic_results = self.store.search(
            question_embedding,
            k=15
        )


        # Keyword search for exact identifiers
        results = semantic_results

        # Combine retrieved chunks
        context = "\n\n".join(
            [
                self.chunks[i]["content"]
                for i in results
            ]
        )


        # Generate response
        response = ollama.chat(

            model="llama3.1",

            messages=[

                {
                    "role": "system",
                    "content":
                    (
                      """You are an AI document assistant.

Answer questions using only the provided document context.

Your goals:
- Retrieve accurate information from the uploaded documents.
- Summarize documents or sections when requested.
- Explain difficult concepts clearly.
- Help users review and understand the material.

When summarizing:
- Identify the main topic.
- Provide a short overview.
- Organize important concepts into sections.
- Include important details, examples, formulas, or processes when available.
- End with concise key takeaways.

When answering questions:
- Be direct and specific.
- Use only the provided context.
- Do not combine unrelated documents or unrelated sections.
- Do not use outside knowledge.

If the answer cannot be found in the provided context, respond exactly:
'I could not find this information in the uploaded documents.'

Always prioritize accuracy over making assumptions.
"""

                    )
                },

                {
                    "role": "user",
                    "content":
                    f"""
Context:
{context}

Question:
{question}
"""
                }
            ]
        )


        # Sources
        sources = []

        for i in results:

            sources.append(
                {
                    "filename": self.chunks[i]["filename"],
                    "page": self.chunks[i]["page"]
                }
            )

            # Sort citations
        sources = sorted(
            sources,
            key=lambda x: (
                x["filename"],
                x["page"]
                )
        )


        return (
            response["message"]["content"],
            sources
        )