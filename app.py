import streamlit as st
import os
from rag import process_pdf, create_rag_chain, get_answer, summarize_document

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

st.title("📚 Enterprise Knowledge Assistant")
st.caption("Chat with your documents using RAG")

# SAFE API HANDLING (LOCAL + CLOUD)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    pass  # already set in environment
else:
    os.environ["GOOGLE_API_KEY"] = "your_local_api_key"
    
# Session
if "chat" not in st.session_state:
    st.session_state.chat = []

if "ready" not in st.session_state:
    st.session_state.ready = False

# Sidebar upload
st.sidebar.header("📂 Upload PDFs")
pdf_files = st.sidebar.file_uploader("Upload", type="pdf", accept_multiple_files=True)

# Process
if pdf_files and not st.session_state.ready:
    with st.spinner("Processing..."):
        paths = []
        for pdf in pdf_files:
            path = f"/tmp/{pdf.name}"
            with open(path, "wb") as f:
                f.write(pdf.read())
            paths.append(path)

        vector_store = process_pdf(paths)
        rag_chain, retriever = create_rag_chain(vector_store)

        st.session_state.vector_store = vector_store
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
        st.session_state.ready = True

        st.sidebar.success("✅ Ready!")

# Buttons
col1, col2 = st.sidebar.columns(2)

if col1.button("📊 Summary"):
    if st.session_state.ready:
        st.sidebar.write(summarize_document(st.session_state.rag_chain))

if col2.button("🗑 Clear"):
    st.session_state.chat = []

# Chat
query = st.chat_input("Ask question...")

if query and st.session_state.ready:
    answer, sources = get_answer(
        st.session_state.rag_chain,
        st.session_state.retriever,
        query
    )

    st.session_state.chat.append((query, answer, sources))

# Display
for q, a, s in st.session_state.chat:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        st.caption("📄 " + ", ".join(s))
