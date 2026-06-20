# 🤖 Zyro Dynamics HR Help Desk Chatbot (RAG)

An AI-powered HR Help Desk chatbot built using **Retrieval-Augmented Generation (RAG)** to answer employee HR policy questions accurately from company documents.

This project was developed as part of the **Zyro Dynamics HR Help Desk RAG Challenge**.

---

## 🚀 Project Overview

The chatbot enables employees to ask HR-related questions in natural language and retrieves accurate answers from HR policy PDFs using a Retrieval-Augmented Generation (RAG) pipeline.

The system combines vector search with Large Language Models (LLMs) to provide reliable, context-aware responses while minimizing hallucinations.

---

## ✨ Features

- 📄 Load multiple HR policy PDF documents
- ✂️ Intelligent document chunking
- 🔍 Semantic search using FAISS Vector Database
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 LangChain LCEL pipeline
- 💬 Interactive Streamlit chatbot
- 📚 Source citation for every response
- 🚫 Guardrails for out-of-scope questions
- 📊 LangSmith tracing support
- ⚡ Fast inference using Groq LLM

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Frameworks & Libraries
- LangChain
- FAISS
- HuggingFace Embeddings
- Streamlit
- LangSmith

### LLM
- Groq API

### Development Platforms
- Kaggle
- Google Colab
- VS Code

---

## 📂 Project Workflow

```
HR Policy PDFs
       │
       ▼
Document Loader
       │
       ▼
Text Chunking
       │
       ▼
Embeddings
       │
       ▼
FAISS Vector Store
       │
       ▼
Retriever
       │
       ▼
Groq LLM
       │
       ▼
Final Answer + Source Citation
```

---

## 📁 Project Structure

```
├── app.py                 # Streamlit Application
├── finalproject.ipynb     # Kaggle/Colab Notebook
├── requirements.txt
├── README.md
├── data/
│   └── HR Policy PDFs
└── submission.csv
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/repository-name.git
```

Move into the project folder

```bash
cd repository-name
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create API keys for:

- GROQ_API_KEY
- LANGCHAIN_API_KEY

Configure them using environment variables or Kaggle Secrets.

---

## ▶️ Running the Project

Run the notebook to generate the vector database.

Then launch the Streamlit application.

```bash
streamlit run app.py
```

---

## 📸 Demo

Add screenshots of:

- Chat Interface
- Question Answering
- Source Citation
- LangSmith Trace

---

## 🎯 Example Questions

- What is the company's leave policy?
- How many casual leaves are available?
- What are the office working hours?
- What is the maternity leave policy?
- What is the probation period?

---

## 📈 Learning Outcomes

Through this project, I gained practical experience in:

- Retrieval-Augmented Generation (RAG)
- LangChain LCEL
- FAISS Vector Search
- Semantic Search
- Prompt Engineering
- Document Chunking
- Embedding Models
- Streamlit Application Development
- LangSmith Tracing
- Production-ready AI Workflow Development

---

## 🔮 Future Improvements

- Multi-language support
- Voice-based HR Assistant
- Authentication for employees
- Chat history
- Feedback and rating system
- Hybrid Search (BM25 + Vector Search)
- Cloud deployment on AWS or Azure

---

## 👨‍💻 Author

**Madan K**

B.Tech Computer Science Engineering Student

- GitHub: https://github.com/MadanTechJourney
- LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Acknowledgements

- LangChain
- Groq
- HuggingFace
- Streamlit
- FAISS
- Kaggle
- Zyro Dynamics HR Help Desk RAG Challenge

---

If you found this project useful, consider giving the repository a ⭐.
