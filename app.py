import os
import streamlit as st
from rag import RAGPipeline

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
page_title="AI Document Assistant",
page_icon="🤖",
layout="wide",
initial_sidebar_state="expanded"
)


# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>

    .block-container{
        padding-top:2rem;
        padding-bottom:1rem;
        max-width:1200px;
    }

    h1,h2,h3{
        font-weight:700;
    }

    .hero{
        padding:30px;
        border-radius:18px;
        background:linear-gradient(135deg,#1f2937,#111827);
        border:1px solid #30363d;
        margin-bottom:25px;
    }

    .hero-title{
        font-size:42px;
        font-weight:800;
        color:white;
        margin-bottom:8px;
    }

    .hero-subtitle{
        color:#c9d1d9;
        font-size:18px;
        margin-bottom:20px;
    }

    .feature-box{
        display:inline-block;
        background:#202938;
        color:white;
        padding:8px 14px;
        margin:4px;
        border-radius:20px;
        font-size:14px;
        border:1px solid #30363d;
    }

    .section-card{
        border:1px solid #30363d;
        border-radius:15px;
        padding:20px;
        margin-top:15px;
        margin-bottom:15px;
    }

    .doc-card{
        border:1px solid #30363d;
        border-radius:12px;
        padding:12px;
        margin-bottom:10px;
        background:#1b1f24;
    }

    .doc-title{
        font-weight:600;
        font-size:16px;
    }

    .small-text{
        color:#9ca3af;
        font-size:13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Hero Section
# ---------------------------------------------------

st.markdown(
"""
<div class="hero">

<div class="hero-title">
🤖 AI Document Assistant
</div>

<div class="hero-subtitle">
Upload documents, retrieve information instantly, summarize content, and ask natural language questions using Retrieval-Augmented Generation (RAG).
</div>

</div>
""",
unsafe_allow_html=True
)


# -------------------- Tabs --------------------

tab1, tab2 = st.tabs(["Model", "About"])

# -------------------- About Model --------------------

with tab1:

    # ---------------------------------------------------
    # Session State
    # ---------------------------------------------------

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents" not in st.session_state:
        st.session_state.documents = []

    # ---------------------------------------------------
    # Upload Documents
    # ---------------------------------------------------

    MAX_FILES = 5

    st.subheader("📚 Uploaded Documents")

    uploaded_files = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=True
    )

    uploaded_count = len(uploaded_files) if uploaded_files else 0

    # Check limit 

    if uploaded_count > MAX_FILES:
            st.error(f"You can upload a maximum of {MAX_FILES} PDFs.")
            st.stop()

    st.progress(uploaded_count / MAX_FILES)
    st.caption(f"**{uploaded_count}/{MAX_FILES}** document(s) uploaded")

    # ---------------------------------------------------
    # Build Knowledge Base
    # ---------------------------------------------------

    if uploaded_files:

        with st.spinner("Uploading..."):

            pipeline = RAGPipeline()

            status = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):

                status.info(
                    f"Processing {i+1} of {uploaded_count}: {uploaded_file.name}"
                )

                os.makedirs(
                    "uploaded_files",
                    exist_ok=True
                )

                pdf_path = os.path.join(
                    "uploaded_files",
                    uploaded_file.name
                )

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                pipeline.add_pdf(pdf_path)

            status.info("Creating vector database...")

            pipeline.build_index()

            st.session_state.pipeline = pipeline

            status.empty()

        st.session_state.documents = []

        for uploaded_file in uploaded_files:

            file_size = uploaded_file.size / (1024 * 1024)

            st.session_state.documents.append(
                {
                    "name": uploaded_file.name,
                    "size": file_size
                }
            )

    st.divider()

    # ---------------------------------------------------
    # Quick Actions
    # ---------------------------------------------------   
    st.subheader("✨ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        summarize = st.button(
            "📝 Summarize Documents",
            use_container_width=True
        )

    with col2:
        key_points = st.button(
            "📌 Key Takeaways",
            use_container_width=True
        )

    with col3:
        explain = st.button(
            "📖 Explain Main Concepts",
            use_container_width=True
        )

    question = None

    if summarize:
        question = "Summarize the uploaded documents."

    elif key_points:
        question = "What are the most important points from the uploaded documents?"

    elif explain:
        question = "Explain the main concepts from the uploaded documents."


    # ---------------------------------------------------
    # Conversation
    # ---------------------------------------------------    
    st.subheader("💬 Chat")

    if len(st.session_state.messages) == 0:
        st.info( "Ask a question or use one of the quick actions above.")

# ---------------------------------------------------
# Message
# ---------------------------------------------------

chat_container = st.container()

with chat_container:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):

            st.markdown(
                message["content"],
                unsafe_allow_html=True)
# ---------------------------------------------------
# Question Input
# ---------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)       

if question:

    if st.session_state.pipeline is None:

        st.warning("Please upload a PDF first.")
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.status(
        "Searching knowledge base...",
        expanded=True
    ) as status:

        status.write("🔎 Searching vector database...")

        answer, sources = (
            st.session_state.pipeline.ask(question)
        )

        status.write("🧠 Generating response...")

        status.update(
            label="Complete",
            state="complete"
        )

    # -----------------------------
    # Build Sources
    # -----------------------------

    grouped_sources = {}

    for source in sources:

        grouped_sources.setdefault(
            source["filename"],
            []
        ).append(source["page"])

    source_html = ""

    for filename, pages in grouped_sources.items():

        pages = sorted(set(pages))

        page_text = ", ".join(
            map(str, pages)
        )

        source_html += f"""
<div class="chat-source">
    <div class="source-title">
        📄 {filename}
    </div>

    Pages: {page_text}
</div>
"""

    final_answer = f"""
<div class="answer-box">

{answer}

</div>

{source_html}
"""

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )

    # Force a rerun so the updated conversation is
    # displayed above the fixed chat input.
    st.rerun()


st.markdown("""
<style>

.main .block-container{

padding-bottom:120px;

}

.stChatInput{

background:#0E1117;

border-top:1px solid #30363d;

}

</style>
""", unsafe_allow_html=True)

# -------------------------
# About
# -------------------------
with tab2:

    st.subheader("Model")

    st.write("**LLM**")
    st.caption("Llama 3.1 (Ollama)")

    st.write("**Embedding Model**")
    st.caption("all-MiniLM-L6-v2")

    st.write("**Vector Database**")
    st.caption("FAISS")

    st.divider()

    st.markdown("""

    ### AI-powered document assistant built with: Python & Streamlit
    
    **Ollama** – Runs large language models locally to generate answers without relying on cloud-based APIs.
    
    **FAISS** – Performs fast similarity searches to retrieve the most relevant document chunks from the knowledge base.
    
    **Hugging Face Transformers** – Provides the pre-trained embedding model that converts text into numerical vectors for semantic search.
    """)
