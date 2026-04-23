# RAG-Based Customer Support Assistant
**Innomatics Research Labs | Final Internship Project**
Kartik Patade | February 2026 Cohort

---

## Project Summary

This is a Retrieval-Augmented Generation (RAG) system built as a customer support assistant. It reads from a PDF knowledge base, retrieves relevant sections using vector similarity search, and uses a language model (Google Gemini) to answer user queries. The workflow is managed using LangGraph. Queries that the system cannot confidently answer are flagged for human review (HITL).

---

## Project Structure

```
.
├── rag_support_assistant.py   # Main application
├── requirements.txt           # Python dependencies
├── knowledge_base.pdf         # Your PDF document (add this yourself)
├── chroma_store/              # Auto-created by ChromaDB on first run
└── escalation_log.txt         # Auto-created when queries are escalated
```

---

## Setup

1. Clone the repo and navigate to the folder.

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set your Google API key:
```
export GOOGLE_API_KEY=your_key_here
```

4. Add your PDF knowledge base file as `knowledge_base.pdf` in the project root.

5. Run the assistant:
```
python rag_support_assistant.py
```

On first run, it will ingest the PDF and build the ChromaDB vector store. Subsequent runs will load the existing store directly.

---

## How It Works

**Ingestion (runs once):**
- PDF is loaded using PyPDFLoader
- Text is split into 500-character chunks with 50-character overlap
- Chunks are embedded using all-MiniLM-L6-v2 (runs locally)
- Embeddings are stored in ChromaDB

**Query flow (every user question):**
1. Query is embedded using the same model
2. Top 3 most similar chunks are retrieved from ChromaDB
3. Chunks + query are sent to Gemini 1.5 Flash as a prompt
4. Response is checked for uncertainty phrases
5. If confident: answer is shown to the user
6. If uncertain or complex: query is flagged and logged (HITL)

---

## Escalation Logic

A query is escalated if:
- No relevant chunks were found in the document
- The LLM response contains phrases like "I cannot find", "not sure", etc.
- The query contains keywords like "legal", "billing dispute", "complaint"

Escalated queries are logged to `escalation_log.txt`.

---

## Notes

- This system is designed for a single PDF knowledge base. Multi-document support is a planned future enhancement.
- Conversation memory is not implemented. Each query is handled independently.
- The HITL escalation in this version simulates human handoff via console output and log file.
