from chromadb import PersistentClient
import os

# Determine the database path
# ChromaDB stores data in a directory, check if chroma.sqlite3 exists in current dir or subdirs
db_path = "."
if os.path.exists("chroma.sqlite3"):
    # If chroma.sqlite3 is in current directory, ChromaDB data might be in a subdirectory
    # Try common locations or use current directory
    db_path = "."
elif os.path.exists("rag_database"):
    db_path = "rag_database"
else:
    db_path = "."

# Connect to ChromaDB
client = PersistentClient(path=db_path)

# List all collections
collections = client.list_collections()
print(f"Found {len(collections)} collection(s):")
for collection in collections:
    print(f"  - {collection.name}")

# Read and print data from each collection
for collection_info in collections:
    collection = client.get_collection(collection_info.name)
    count = collection.count()
    print(f"\n{'='*60}")
    print(f"Collection: {collection_info.name}")
    print(f"Total items: {count}")
    print(f"{'='*60}")
    
    # Get all data from the collection
    results = collection.get()
    
    if results and 'ids' in results and len(results['ids']) > 0:
        print(f"\nFirst 5 items (showing up to 5):")
        for i in range(min(5, len(results['ids']))):
            print(f"\n--- Item {i+1} ---")
            print(f"ID: {results['ids'][i]}")
            if 'documents' in results and results['documents']:
                print(f"Document: {results['documents'][i][:200]}..." if len(results['documents'][i]) > 200 else f"Document: {results['documents'][i]}")
            if 'metadatas' in results and results['metadatas']:
                print(f"Metadata: {results['metadatas'][i]}")
            if 'embeddings' in results and results['embeddings']:
                embedding = results['embeddings'][i]
                print(f"Embedding dimension: {len(embedding)}")
                print(f"Embedding (first 5 values): {embedding[:5]}")
    else:
        print("No data found in this collection.")

