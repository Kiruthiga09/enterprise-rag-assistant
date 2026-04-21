import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ------------------- PROCESS PDF -------------------
def process_pdf(pdf_files):
    documents = []
    skipped_files = []

    for pdf in pdf_files:
        doc = fitz.open(pdf)
        total_pages = doc.page_count
        doc.close()

        if total_pages > 50:
            skipped_files.append(os.path.basename(pdf))
            continue

        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(pdf)

        documents.extend(docs)

    if not documents:
        raise ValueError("⚠️ All uploaded PDFs exceed 50 pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip() != ""]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # 🔥 ADDED ERROR HANDLING HERE (embedding)
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise ValueError("⚠️ Embedding API limit reached. Please wait and try again.")
        else:
            raise ValueError(f"⚠️ Error: {str(e)}")

    return vector_store, skipped_files


# ------------------- CREATE RAG -------------------
def create_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent document assistant.

    STRICT RULES:
    - Answer ONLY using the provided context
    - If partial info exists, give best possible answer
    - Use bullet points if helpful

    Context:
    {context}

    Question:
    {question}
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, retriever


# ------------------- GET ANSWER -------------------
def get_answer(chain, retriever, query, selected_file=None):
    docs = retriever.invoke(query)

    if selected_file:
        docs = [d for d in docs if d.metadata.get("source") == selected_file]

    if not docs:
        return "Not available in selected document.", []

    # 🔥 ADDED ERROR HANDLING
    try:
        response = chain.invoke(query)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ API limit reached. Please wait and try again.", []
        else:
            return f"⚠️ Error: {str(e)}", []

    try:
        if isinstance(response.content, list):
            answer = response.content[0]["text"]
        else:
            answer = response.content
    except:
        answer = str(response)

    sources = list(set([
        f"{d.metadata.get('source')} - Page {d.metadata.get('page', '?')}"
        for d in docs
    ]))

    return answer, sources


#---------------------summary---------------------
def summarize_document(vector_store, selected_file=None):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.2
    )

    docs = list(vector_store.docstore._dict.values())

    if selected_file:
        docs = [d for d in docs if d.metadata.get("source") == selected_file]

    if not docs:
        return "⚠️ No content available for selected document."

    text = "\n\n".join([doc.page_content for doc in docs[:20]])

    prompt = f"""
    Summarize the following document clearly in bullet points:

    {text}
    """

    # 🔥 ADDED ERROR HANDLING
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ API limit reached. Please wait and try again."
        else:
            return f"⚠️ Error: {str(e)}"

    try:
        if isinstance(response.content, list):
            return response.content[0]["text"]
        return response.content
    except:
        return str(response)
