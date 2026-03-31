import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Mocks for the Command Center API

IT_KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "Azure AD Account Lockout",
        "content": "If a user is locked out of Azure AD due to too many failed login attempts, the Helpdesk should run the `Unlock-AzureADUser` runbook. Ensure you verify the user's identity via Okta MFA push before executing."
    },
    {
        "id": "KB-002",
        "title": "High CPU Usage on Corporate Laptops",
        "content": "If a corporate laptop (Windows 11) is reporting CPU utilization >95% for more than 10 minutes, the root cause is often the CrowdStrike Falcon sensor or an interrupted Windows Update. Action: Run a remote diagnostic script to check top consuming processes."
    },
    {
        "id": "KB-003",
        "title": "VPN Gateway Issues (Stockholm Office)",
        "content": "Users in the Stockholm office experiencing dropped VPN connections should be routed to the secondary gateway endpoint `sto-sec.vpn.corp.local`. Do not restart the primary router without Level 3 approval."
    }
]

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        _vector_store = Chroma(
            embedding_function=embeddings,
            collection_name="it_knowledge_base"
        )
        
        # Auto-seed the database if it is empty (e.g., on first boot in production)
        if _vector_store._collection.count() == 0:
            print("Initializing Vector DB: Empty database detected. Auto-seeding...")
            texts = [doc["content"] for doc in IT_KNOWLEDGE_BASE]
            metadatas = [{"title": doc["title"], "id": doc["id"]} for doc in IT_KNOWLEDGE_BASE]
            ids = [doc["id"] for doc in IT_KNOWLEDGE_BASE]
            _vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print("Initialization Complete: Added IT Knowledge Base to Vector DB.")
            
    return _vector_store

def search_knowledge_base(query: str) -> str:
    """Performs a semantic search over the IT knowledge base using Chroma."""
    try:
        vs = get_vector_store()
        # Retrieve the top 2 most relevant documents
        docs = vs.similarity_search(query, k=2)
        
        if not docs:
            raise ValueError("No relevant documentation found in the IT Knowledge Base.")
            
        results = []
        for doc in docs:
            results.append(doc.page_content)
            
        return "\n\n---\n\n".join(results)
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        # Raise the exception so the LangGraph agent can catch it, log it, and trigger a fallback
        raise e
