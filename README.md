# 📚 Enterprise Knowledge Assistant (RAG)

This project is a Streamlit-based application that allows users to chat with PDF documents using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Upload multiple PDF documents
- Ask questions based on document content
- Generate summaries of selected documents
- Handles large files (skips PDFs > 50 pages)
- Displays source references for answers

## 🛠 Technologies Used
- Python
- Streamlit
- LangChain
- FAISS (Vector Store)
- Google Generative AI (Gemini)
- PyMuPDF

## 📂 Project Structure
```
app.py        # Streamlit frontend
rag.py        # RAG pipeline logic
requirements.txt
```

## 🔑 Setup Instructions

1. Clone the repository:
```
git clone https://github.com/Kiruthiga09/Enterprise_Knowledge_Assistant
cd <repo-folder>
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Add API Key in Streamlit Secrets:
```
GOOGLE_API_KEY = "your_api_key"
```

4. Run the app:
```
streamlit run app.py
```

## ⚠️ Notes
- PDFs with more than 50 pages are skipped
- Ensure correct API key is set
- If API limit is reached, wait and retry

## 📌 Future Improvements
- Multi-document comparison
- Better UI/UX
- Support for other file formats

## 👨‍💻 Author
Kiruthiga K
