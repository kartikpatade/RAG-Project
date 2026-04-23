

import os
from datetime import datetime
from typing import TypedDict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- Config ---
PDF_PATH = "knowledge_base.pdf"
CHROMA_DIR = "./chroma_store"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAyvDreJoxD7KNbNV7rWb2j847Nh2lYmfI")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

UNCERTAINTY_PHRASES = [
    "i don't know",
    "i cannot find",
    "cannot answer",
    "not sure",
    "no information",
    "not mentioned",
    "outside the scope",
]

COMPLEX_KEYWORDS = ["legal", "billing dispute", "refund claim", "complaint", "sue", "lawsuit"]


# --- Graph State ---
class GraphState(TypedDict):
    query: str
    context: str
    response: str
    escalate: bool
    retrieved_docs: List[str]


# --- Ingestion ---
def ingest_pdf(pdf_path: str, chroma_dir: str, embeddings):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_dir)
    #vectorstore.persist() 
    print("Embeddings stored in ChromaDB.")
    return vectorstore


def load_vectorstore(chroma_dir: str, embeddings):
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)


# --- Routing logic ---
def should_escalate(response: str, retrieved_docs: list, query: str) -> bool:
    if not retrieved_docs:
        return True

    response_lower = response.lower()
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in response_lower:
            return True

    query_lower = query.lower()
    for keyword in COMPLEX_KEYWORDS:
        if keyword in query_lower:
            return True

    return False


# --- HITL handler ---
def handle_hitl(state: GraphState):
    print("\n[ESCALATED] This query has been flagged for human review.")
    print(f"Query: {state['query']}")
    print(f"Attempted response: {state['response']}")
    print("A support agent will follow up shortly.\n")

    with open("escalation_log.txt", "a") as f:
        f.write(f"{datetime.now()} | QUERY: {state['query']} | RESPONSE: {state['response']}\n")


def fallback_from_context(retrieved_texts: List[str]) -> str:
    if not retrieved_texts:
        return "I cannot find that information in our documentation."

    top_chunk = retrieved_texts[0].strip().replace("\n", " ")
    if len(top_chunk) > 500:
        top_chunk = top_chunk[:500].rstrip() + "..."
    return f"Based on the available documentation: {top_chunk}"


# --- LangGraph Nodes ---
def retrieve_and_generate(state: GraphState) -> GraphState:
    query = state["query"]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    retrieved_texts = [doc.page_content for doc in docs]
    context = "\n\n".join(retrieved_texts)

    prompt = (
        "You are a helpful customer support assistant.\n"
        "Use ONLY the context below to answer the question.\n"
        "If the context does not contain enough information to answer, say 'I cannot find that information in our documentation.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    try:
        llm_response = llm.invoke(prompt)
        response_text = llm_response.content
    except Exception:
        response_text = fallback_from_context(retrieved_texts)

    return {
        **state,
        "context": context,
        "response": response_text,
        "retrieved_docs": retrieved_texts,
    }


def route_response(state: GraphState) -> GraphState:
    escalate = should_escalate(state["response"], state["retrieved_docs"], state["query"])
    return {**state, "escalate": escalate}


def output_node(state: GraphState) -> GraphState:
    if state["escalate"]:
        handle_hitl(state)
    else:
        print(f"\nAnswer: {state['response']}\n")
    return state


def route_after_response(state: GraphState) -> str:
    return "output_node"


# --- Build Graph ---
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve_and_generate", retrieve_and_generate)
    graph.add_node("route_response", route_response)
    graph.add_node("output_node", output_node)

    graph.set_entry_point("retrieve_and_generate")
    graph.add_edge("retrieve_and_generate", "route_response")
    graph.add_edge("route_response", "output_node")
    graph.add_edge("output_node", END)

    return graph.compile()


# --- Main ---
if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        if not os.path.exists(PDF_PATH):
            print(f"Error: PDF file '{PDF_PATH}' not found. Please add a knowledge base PDF.")
            exit(1)
        vectorstore = ingest_pdf(PDF_PATH, CHROMA_DIR, embeddings)
    else:
        print("Loading existing vector store...")
        vectorstore = load_vectorstore(CHROMA_DIR, embeddings)

    app = build_graph()

    print("\nCustomer Support Assistant is ready. Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            print("Goodbye.")
            break
        if not query:
            continue

        initial_state = GraphState(
            query=query,
            context="",
            response="",
            escalate=False,
            retrieved_docs=[],
        )

        app.invoke(initial_state)
