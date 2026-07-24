from sentence_transformers import SentenceTransformer
from document_processor import DocumentProcessor
from vector_store import VectorStore
import streamlit as st
from groq import Groq



class RAGPipeline:

    def __init__(self):
        
        self.client = Groq(
            api_key=st.secrets["GROQ_API_KEY"]
        )

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

        question_embedding = self.model.encode([question])

        results = self.store.search(
            question_embedding,
            k=15
        )

        context = "\n\n".join(
            self.chunks[i]["content"]
            for i in results
        )

        system_prompt = """
        You are an AI document assistant.

        Answer questions only using the provided context.

        If the answer cannot be found in the uploaded documents,
        say:
        "I could not find this information in the uploaded documents."
        """

        response = self.client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            temperature=0.2,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": f"""
    Context:
    {context}

    Question:
    {question}
    """
                }
            ]
        )

        sources = []

        for i in results:

            sources.append(
                {
                    "filename": self.chunks[i]["filename"],
                    "page": self.chunks[i]["page"]
                }
            )

        sources = sorted(
            sources,
            key=lambda x: (
                x["filename"],
                x["page"]
            )
        )

        return (
            response.choices[0].message.content,
            sources
        )