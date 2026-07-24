# AI-Document-Assistant

Live Demo: https://ai-document-assistant-9iljvahcqcl6m6pevgnzzr.streamlit.app

## Project Goal

The goal of this project is to help users quickly understand and search through PDF documents using artificial intelligence. Instead of manually reading hundreds of pages, users can upload one or more PDFs and ask questions in plain English. 

The assistant retrieves the most relevant information from the uploaded documents and generates clear, easy-to-read answers with page references so users can verify the source.


## How It Was Built

The application follows a Retrieval-Augmented Generation (RAG) workflow.

1. Users upload one or more PDF documents.
2. The application extracts the text from each page.
3. The text is divided into smaller sections (chunks) to improve search accuracy.
4. Each chunk is converted into a numerical representation (embedding) that captures its meaning.
5. All embeddings are stored in a FAISS vector database for fast semantic searching.
6. When a user asks a question, the application searches for the most relevant document sections instead of scanning every page.
7. The retrieved information is sent to a Large Language Model (LLM), which generates a natural-language answer based only on the uploaded documents.
8. The application displays both the answer and the document pages used to generate the response, allowing users to verify the information.

| Tool                         | Purpose                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Python**                   | Core programming language used to build the application.                                                |
| **NumPy**                    | Handles numerical data used when creating and searching embeddings.                                     |
| **Streamlit**                | Creates the interactive web interface where users upload documents and chat with the AI assistant.      |
| **Groq API**                 | Provides access to a LLM (Llama 3.3) that generates responses from the retrieved document.              |
| **Sentence Transformers**    | Converts document text into embeddings so the application can search by meaning rather than exact words.|
| **FAISS**                    | Stores & searches document embeddings to quickly retrieve the most relevant sections for each question. |
| **PyMuPDF**                  | Extracts text from uploaded PDF documents page by page.                                                 |
| **LangChain Text Splitters** | Breaks large documents into smaller, overlapping chunks to improve retrieval accuracy.                  |
