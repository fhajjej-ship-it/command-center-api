from mocks import get_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

print(f"Chroma DB exists: {os.path.exists('./.chroma_db')}")
vs = get_vector_store()
count = vs._collection.count()
print(f"Collection count: {count}")

try:
    docs = vs.similarity_search("vpn issues", k=2)
    print(f"Found {len(docs)} documents.")
except Exception as e:
    print(f"Error searching: {e}")
