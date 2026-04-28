from data_store import Chroma, get_file_chunks
from langchain.schema import Document
from embedding import get_embedding
from tqdm import tqdm

class DocLoader():
    def __init__(self, chroma_instance: Chroma, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chroma = chroma_instance
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, path: str):
        chunks = self.chunk_document(path)
        self.insert_data_to_chroma(chunks)
        print("Data added successfully")
    
    def chunk_document(self, path: str):
        chunks: list[Document] = get_file_chunks(path, self.chunk_size, self.chunk_overlap)
        return chunks
    
    def insert_data_to_chroma(self, chunks: list[Document]):
        data = []
        count = 0
        for index, chunk in tqdm(enumerate(chunks)):
            chunk_data={
                "embedding": get_embedding(chunk.page_content),
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "id": index
            }
            data.append(chunk_data)
            count += 1

            if count % 20 == 0:
        
                embeddings = [chunk['embedding'] for chunk in data]
                contents = [chunk['content'] for chunk in data]
                metadatas = [chunk['metadata'] for chunk in data]
                ids = [str(chunk['id']) for chunk in data]

                chroma_data = {
                    "embeddings": embeddings,
                    "contents": contents,
                    "metadatas": metadatas,
                    "ids": ids,
                }

                try:
                    self.chroma.add_data_to(
                        data=chroma_data
                    )
                except Exception as e:
                    print(e)
                data = []
        if data:
            embeddings = [chunk['embedding'] for chunk in data]
            contents = [chunk['content'] for chunk in data]
            metadatas = [chunk['metadata'] for chunk in data]
            ids = [str(chunk['id']) for chunk in data]

            chroma_data = {
                "embeddings": embeddings,
                "contents": contents,
                "metadatas": metadatas,
                "ids": ids,
            }

            try:
                self.chroma.add_data_to(
                    data=chroma_data
                )
            except Exception as e:
                print(e)
