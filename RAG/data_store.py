from chromadb import PersistentClient, ClientAPI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create vector store for embeddings
class Chroma():
    """
    Chroma class to instantiate vector db in memory
    """

    def __init__(self, default_database_path: str = "/data/sandesh/llm_ood/rag/chroma_data_store", first_collection_name: str = "qa_collection", top_k: int = 1):
        self.api = PersistentClient(default_database_path)
        if not self.api.count_collections():
            self.collection_pointer = self.api.create_collection(first_collection_name)
        else:
            self.collection_pointer = self.api.get_collection(first_collection_name)
        self.top_k = top_k

    def new_collection(self ,name: str, **kwargs):
        try:
            self.api.create_collection(name, **kwargs)
        except Exception as e:
            print(e)

    def add_data_to(self, data):
        try:
            self.collection_pointer.add(
                embeddings=data.get("embeddings"),
                documents=data.get("contents"),
                metadatas=data.get("metadatas"),
                ids=data.get("ids")
            )
        except Exception as e:
            print(e)
    
    def switch_collection(self, new_pointer: str):
        try:
            self.collection_pointer = self.api.get_collection(new_pointer)
        except Exception as e:
            print(e)
    
    def query(self, embedding: list[float], **kwargs):
        try:
            result = self.collection_pointer.query(query_embeddings=embedding, n_results=self.top_k, **kwargs)
            # print(result)
            return result
        except Exception as e:
            print(e)


# Preparing documents for RAG

def load(loader: TextLoader, chunking, overlap, **kwargs):
    return loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunking, chunk_overlap=overlap),**kwargs)

def extract_text_chunks(file,chunking,overlap,**kwargs):
    loader = TextLoader(file, 'utf-8')
    return load(loader,chunking, overlap, **kwargs)

def get_file_chunks(filename: str, chunk_size: int = 512, chunk_overlap: int = 64):
    try:
        chunks = extract_text_chunks(filename, chunk_size, chunk_overlap)
        return chunks
    except Exception as e:
        print("Error getting chunks from file: ",e)

    


