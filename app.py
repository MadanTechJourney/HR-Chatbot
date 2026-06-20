# ─── Zyro Dynamics HR Help Desk — Streamlit Chatbot ──────────────────────────
# Deploy at: https://share.streamlit.io
# GitHub repo structure:
#   app.py
#   requirements.txt
#   hr_docs/   ← all 11 PDF files go here

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #0f3460;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        font-size: 0.82rem;
        color: #333;
    }
    .oos-badge {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #856404;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .in-scope-badge {
        background: #d1e7dd;
        border: 1px solid #0f5132;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #0f5132;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>Ask any question about HR policies, leave, benefits, conduct, and more.</p>
</div>
""", unsafe_allow_html=True)

# ─── Config ──────────────────────────────────────────────────────────────────
CORPUS_PATH   = "hr_docs/"        # folder containing all 11 PDFs
LLM_MODEL = "llama-3.3-70b-versatile" # Groq model
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions related to Zyro Dynamics HR policies "
    "and employee guidelines. This question falls outside the scope of the HR documents "
    "I have access to. For other queries, please contact the appropriate team or "
    "reach out to hr.helpdesk@zyrodyn amics.com."
)

# ─── Load secrets ────────────────────────────────────────────────────────────
GROQ_API_KEY      = st.secrets.get("GROQ_API_KEY",      os.environ.get("GROQ_API_KEY", ""))
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY", ""))

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

# ─── RAG pipeline (cached — only built once per session) ─────────────────────
@st.cache_resource(show_spinner="🔄 Loading HR policy knowledge base...")
def build_pipeline():
    # Load all PDFs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs   = loader.load()

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embed (local, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # FAISS vector store + MMR retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.6}
    )

    # LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
    )

    return retriever, llm


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        src  = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {src}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Prompts ─────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY using the HR policy document context provided below.
Be concise, precise, and professional. Do not use outside knowledge.
If the context is insufficient, say so and recommend contacting hr.helpdesk@zyrodyn amics.com.

CONTEXT FROM HR POLICY DOCUMENTS:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict classifier for an HR helpdesk.
Determine if the question is about Zyro Dynamics HR policies (leave, payroll, benefits, WFH, conduct, performance, onboarding, separation, travel expenses, POSH, IT/data security).
Reply with EXACTLY one word — either IN_SCOPE or OUT_OF_SCOPE. No explanation."""),
    ("human", "{question}")
])


@traceable(name="streamlit_ask_bot")
def ask_bot(question: str, retriever, llm) -> dict:
    """Full chatbot pipeline with guardrails."""
    # Step 1: Classify question
    clf_prompt   = OOS_PROMPT.invoke({"question": question})
    try:
    response = llm.invoke(clf_prompt)

    if hasattr(response, "content"):
        clf_response = response.content.strip().upper()
    else:
        clf_response = str(response).strip().upper()

except Exception as e:
    st.error(f"Groq Error: {e}")
    raise


    if "OUT_OF_SCOPE" in clf_response:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "classification": "OUT_OF_SCOPE"
        }

    # Step 2: RAG pipeline
    docs    = retriever.invoke(question)
    context = format_docs(docs)
    prompt  = RAG_PROMPT.invoke({"context": context, "question": question})
    answer  = StrOutputParser().invoke(llm.invoke(prompt))

    sources = [
        {
            "source":  os.path.basename(d.metadata.get("source", "?")),
            "page":    d.metadata.get("page", "?"),
            "snippet": d.page_content[:200]
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources, "classification": "IN_SCOPE"}


# ─── Chat session state ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Build pipeline ───────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please add it in Streamlit Secrets → Settings → Secrets.")
    st.stop()

retriever, llm = build_pipeline()
st.success("✅ HR knowledge base ready. Ask me anything about Zyro Dynamics policies!")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Topics I Can Help With")
    topics = [
        "🏖️ Leave Policy (EL, CL, SL, Maternity, Paternity)",
        "🏠 Work From Home Policy",
        "💰 Compensation & Benefits",
        "📈 Performance Reviews & PIP",
        "🛡️ Code of Conduct & Ethics",
        "🔒 IT & Data Security Policy",
        "✈️ Travel & Expense Policy",
        "⚖️ POSH / Sexual Harassment Policy",
        "🚀 Onboarding & Separation",
        "🏢 Company Profile & Grades",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    st.markdown("**📧 HR Contacts**")
    st.markdown("General: hr.helpdesk@zyrodyn amics.com")
    st.markdown("Payroll: payroll@zyrodyn amics.com")
    st.markdown("POSH: icc@zyrodyn amics.com")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─── Display existing chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Policy Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">📑 <b>{s["source"]}</b> — Page {s["page"]}<br>'
                        f'{s["snippet"]}...</div>',
                        unsafe_allow_html=True
                    )

# ─── Chat input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask an HR question (e.g. How many sick leaves do I get?)"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching HR policies..."):
            result = ask_bot(user_input, retriever, llm)

        classification = result.get("classification", "IN_SCOPE")

        if classification == "OUT_OF_SCOPE":
            st.markdown('<span class="oos-badge">⚠️ Out of Scope</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="in-scope-badge">✅ HR Policy Answer</span>', unsafe_allow_html=True)

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander(f"📄 Policy Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f'<div class="source-box">📑 <b>{s["source"]}</b> — Page {s["page"]}<br>'
                        f'{s["snippet"]}...</div>',
                        unsafe_allow_html=True
                    )

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })
