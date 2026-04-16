import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def process_pdf(pdf_files):
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer clearly using ONLY the context.
    If not found, say "Not available in document".

    Context:
    {context}

    Question:
    {question}
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, retriever


def get_answer(chain, retriever, query):
    docs = retriever.invoke(query)

    response = chain.invoke(query)

    try:
        answer = response.content[0]["text"]
    except:
        answer = response.content

    sources = list(set([f"Page {doc.metadata.get('page', '?')}" for doc in docs]))

    return answer, sources


def summarize_document(chain):
    response = chain.invoke("Summarize the document in bullet points")

    try:
        return response.content[0]["text"]
    except:
        return response.content
