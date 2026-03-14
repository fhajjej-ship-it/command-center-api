import os
from dotenv import load_dotenv
from typing import List, Dict

# Chroma and LangChain imports
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import the raw knowledge base data to seed
from mocks import IT_KNOWLEDGE_BASE

def seed_database(db_path: str = "./.chroma_db"):
    """
    Seeds the local Chroma database with mock IT Knowledge Base entries using Gemini embeddings.
    """
    load_dotenv()
    
    # Ensure the Google API Key is present
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is missing. Cannot generate embeddings.")

    print(f"Connecting to Chroma database at {db_path}...")
    
    # Initialize the embedding model
    # We use Google's gemini-embedding-001 model for fast, high-quality embeddings.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Extract texts and metadatas for LangChain's Chroma integration
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []
    
    for doc in IT_KNOWLEDGE_BASE:
        # Combining title and content gives the embedding model more context
        full_text = f"Title: {doc['title']}\nContent: {doc['content']}"
        texts.append(full_text)
        metadatas.append({"id": doc["id"], "title": doc["title"]})
        ids.append(doc["id"])
        
    print(f"Generating embeddings for {len(texts)} documents and inserting into Chroma...")
    
    # This will create (or update) the vector store on disk
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=db_path,
        collection_name="it_knowledge_base"
    )
    
    print("Database seeded successfully!")
    print(f"Total documents in collection layout: {vector_store._collection.count()}")

if __name__ == "__main__":
    seed_database()
